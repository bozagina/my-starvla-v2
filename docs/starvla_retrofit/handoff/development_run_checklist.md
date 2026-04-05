# StarVLA Retrofit 每轮开发清单（可直接执行）

用途：统一“每次开发前/中/后”的动作，避免流程偏离与验收死锁。

---

## 1) 每次开发前（必须先做）

1. 进入仓库：

```bash
cd /Users/bazinga/code/my-starvla-v2
```

2. 一键 readiness（是否可开工）：

```bash
bash /Users/bazinga/code/my-starvla-v2/tools/handoff/pre_dev_readiness.sh
```

通过标准：输出 `READY_TO_DEVELOP=YES`。

3. 打印本流启动 prompt（给新会话/新线程）：

```bash
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh prompt-retrofit
```

4. 创建本轮 EXP_ID（首个代码改动前）：

```bash
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start \
  --module INFRA \
  --owner OC \
  --title "<one line task title>"
```

5. 分支检查（确保在 `codex/tmp-*`）：

```bash
git -C /Users/bazinga/code/my-starvla-v2 rev-parse --abbrev-ref HEAD
```

6. 并行线程开发时，先选线程文档再动手：
- `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
- `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/threads/`

---

## 2) 每轮推荐 Prompt（复制即用）

```text
请执行 StarVLA retrofit 开发流程，并严格遵守：

1) 先读取：
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/context_pack_compact.md
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/acceptance_deadlock_guard.md
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md（仅尾部最近记录）

2) 输出不超过8条当前状态与约束。
   - 并明确：当前仓库仅使用 Qwen2.5VL / Qwen3VL；MapAnything/LLaVA3D 已解耦，不作为当前实现依赖。

3) 先执行并报告：
   - bash /Users/bazinga/code/my-starvla-v2/tools/handoff/pre_dev_readiness.sh
   - bash /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh

4) 在首个代码改动前创建 EXP_ID：
   - /Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start --module INFRA --owner OC --title "<task>"

5) 严格按 gate 执行（P0->P1->P2->P3），不得跳 gate。

6) 每轮输出必须包含：
   - 修改了什么
   - 证据是什么
   - 得到什么结论
   - 下一步建议
   - 建议 commit message（[EXP_ID] one-line intent）

7) 若远程证据缺失，不要无限 IN_PROGRESS，改为 BLOCKED_WAIT_REMOTE 并列出缺失工件。
```

---

## 3) 开发中（每次提交前）

1. 拓扑与工作区检查：

```bash
bash /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh
```

2. 死锁风险检查（状态是否悬挂）：

```bash
python /Users/bazinga/code/my-starvla-v2/tools/handoff/check_deadlock_risk.py --max-open-hours 24
```

3. 重大修改后追加日志：
- `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md`

---

## 4) 远程验证相关（需要时）

1. 拉取远程 run 证据包：

```bash
bash /Users/bazinga/code/my-starvla-v2/tools/fetch_latest_run_files.sh
```

2. 验收最小证据集（缺一不可）：
- `run_identity.txt`
- `config.yaml`
- `metrics.jsonl`
- `summary.jsonl`
- `train.log` 或 `train.raw.log`

3. Shared builder 样本契约校验（P2 起推荐执行）：

Schema 文档：
- `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/shared_builder_schema_contract.md`

Validator CLI：
- `/Users/bazinga/code/my-starvla-v2/tools/handoff/validate_shared_builder_schema.py`

示例：

```bash
python /Users/bazinga/code/my-starvla-v2/tools/handoff/validate_shared_builder_schema.py \
  --smoke-log /path/to/shared_builder_smoke.log \
  --mode auto \
  --expected-schema-version p1_shared_builder_v1 \
  --expected-action-chunk-len 16 \
  --expected-action-dim 7 \
  --require-state
```

---

## 5) 注意事项（高频踩坑）

1. 本地改动不等于远程生效；远程结论只能依据回传日志/配置/指标。
2. 验收以单一硬门为先：`Success@LIBERO`，再看 `p95 latency` 与稳定性；代理指标只做辅证。
3. 同一 `EXP_ID` 多条记录时，以最新状态为准。
4. 工作区若 dirty，提交时必须严格控范围，避免混入无关改动。
5. 当前 gate 未闭环，不要提前进入下一 gate 或 Path-A 扩展。
6. 若发现提示词/白名单/脚本仍引用 MapAnything/LLaVA3D，先清理入口依赖再继续开发。
