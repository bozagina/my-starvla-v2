# StarVLA Retrofit 线程 Prompt 与 Checklist 索引

- EXP_ID: `ALG1-INFRA-20260403-004-OC`
- 目的: 在不做大重构的前提下，把并行开发拆成 4 个可独立执行线程，且避免验收死锁。
- 当前 VLM 范围: 本仓线程默认仅允许 `Qwen2.5VL / Qwen3VL` 路径。

## 1) 通用启动命令（所有线程一致）

```bash
cd /Users/bazinga/code/my-starvla-v2
bash /Users/bazinga/code/my-starvla-v2/tools/handoff/pre_dev_readiness.sh
```

```bash
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start \
  --module <DATA|INFRA|EVAL> \
  --owner OC \
  --title "<thread short title>"
```

必须确认:
1. 当前分支在 `codex/tmp-*`。
2. 三层流转保持 `tmp -> worktree -> final`。
3. 每轮输出都包含: 修改/证据/结论/下一步/commit message。

## 2) 线程分工总览

| 线程 | 文档 | 模块 | 目标 | 主要改动白名单 |
|---|---|---|---|---|
| T1-DATA | `thread_t1_data_builder.md` | DATA | 共享 builder 与窗口抽样接口落位 | `starVLA/dataloader/**`, `starVLA/config/training/*.yaml` |
| T2-SCHEMA | `thread_t2_schema_infra.md` | INFRA | 样本 schema 与 validator | `tools/handoff/**`, `docs/starvla_retrofit/handoff/**` |
| T3-EVAL | `thread_t3_eval_diag_acceptance.md` | EVAL | 验收链路与指标对齐 | `examples/LIBERO/eval_files/**`, `tools/fetch_latest_run_files.sh`, `docs/**` |
| T4-TRAINER | `thread_t4_trainer_insertion.md` | INFRA | trainer 最小插入点(A/corrective) | `starVLA/training/train_starvla.py`, `starVLA/model/framework/Qwen*.py`, `starVLA/model/modules/vlm/QWen2_5.py`, `starVLA/model/modules/vlm/QWen3.py`, `starVLA/config/training/*.yaml` |

## 3) 线程间握手机制（防死锁）

1. T2/T4 如依赖 T1 输出，先在 `progress_live.md` 写明“所需字段/文件/时间点”。
2. 上游未在约定时间交付时，下游不得无限 `IN_PROGRESS`，改 `BLOCKED_WAIT_REMOTE` 并写清缺失工件。
3. 验收线程(T3)只依据可核验证据（`config.yaml/metrics.jsonl/summary.jsonl/train.log` + eval 结果）给结论。
4. 任一线程 DONE 后，必须在 `progress_live.md` 追加“可供下游消费的产物清单”。
5. gate 仍按 `P0 -> P1 -> P2 -> P3`，并行仅限同 gate 内部拆分。

## 4) 建议执行顺序

1. T1 与 T4 可先并行审计与草拟接口。
2. T2 在 T1 输出字段草案后立即固化 schema/validator。
3. T3 持续跟进日志口径与验收冲突规则，等待远程证据后再封板。

## 5) 线程文档入口

1. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t1_data_builder.md`
2. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t2_schema_infra.md`
3. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t3_eval_diag_acceptance.md`
4. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t4_trainer_insertion.md`
