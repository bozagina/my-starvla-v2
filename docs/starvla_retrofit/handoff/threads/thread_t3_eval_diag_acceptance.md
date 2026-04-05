# T3(EVAL): 验收与诊断线程

- 线程定位: 对齐“验收系统 vs 验收指标”的执行口径，避免线程互相等待。
- 建议模块: `EVAL`

## 1) 启动命令

```bash
cd /Users/bazinga/code/my-starvla-v2
bash /Users/bazinga/code/my-starvla-v2/tools/handoff/pre_dev_readiness.sh
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start \
  --module EVAL \
  --owner OC \
  --title "T3 acceptance alignment and diagnostics"
```

## 2) 必读上下文（按顺序）

1. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/context_pack_compact.md`
2. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
3. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/acceptance_deadlock_guard.md`
4. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
5. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/thread_t3_eval_diag_acceptance.md`

## 3) 白名单文件

1. `/Users/bazinga/code/my-starvla-v2/examples/LIBERO/eval_files/eval_libero.py`
2. `/Users/bazinga/code/my-starvla-v2/examples/LIBERO/eval_files/model2libero_interface.py`
3. `/Users/bazinga/code/my-starvla-v2/tools/fetch_latest_run_files.sh`
4. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/*.md`

## 4) 线程专用 Prompt（复制即用）

```text
你是 T3(EVAL) 线程，目标是统一验收链路与指标定义，防止“验收线程等开发线程”死锁。

要求:
1) 先读取 acceptance_deadlock_guard + thread index。
2) 本地只做 eval/diagnostic 代码与文档，不对远程效果做未经证据的结论。
3) 验收优先级:
   - 硬门: Success@LIBERO
   - 软门: latency、稳定性、代理指标
4) 若缺远程工件，状态必须为 BLOCKED_WAIT_REMOTE，并列缺失工件列表。
5) 每轮输出必须包含: 修改/证据/结论/下一步/commit message。
```

## 5) Upstream Pass Checks

1. T1/T2 至少有字段约定（否则仅输出验收口径文档，不给通过结论）。
2. 远程结果判定必须有最小证据集: `config.yaml + metrics.jsonl + summary.jsonl + train.log`。
3. 与 trainer 线程(T4)约定新增 metrics key 名称，避免取数失败。

## 6) Checklist

1. 固化 eval 结果日志口径（成功率、episode 数、失败原因）。
2. 固化 chunk 推理口径（`action_chunk_size = future_action_window_size + 1`）。
3. 定义“指标冲突时”裁决规则并写入文档。
4. 增加 deadlock guard 触发条件和处理动作。

## 7) Done 证据格式

1. 命令证据: 拉取与解析日志命令可复现。
2. 文档证据: 明确硬门/软门优先级。
3. `progress_live.md` 必须写清“当前结论是本地结论还是远程结论”。

## 8) 交接规则

1. 给 T1/T2: 验收所需字段最小集合。
2. 给 T4: trainer 必须写出的 metrics key 列表。
3. 缺远程日志时不等待无限循环，直接 `BLOCKED_WAIT_REMOTE`。
