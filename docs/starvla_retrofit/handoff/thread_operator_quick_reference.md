# Harness v2 线程操作者快速参考

本文档是六线程并行开发的 **唯一操作入口**。每次启动新线程时，按下面流程执行。

---

## 通用启动流程（所有线程共用）

```bash
# ① 设置环境变量
export STARVLA_EXPECTED_REPO_ROOT="$(git rev-parse --show-toplevel)"
export STARVLA_EXPECTED_VLM_SCOPE="qwen_only"
export STARVLA_THREAD_ID="<从下表选一个>"

# ② 运行 readiness（含 v2 线程校验）
bash "$STARVLA_EXPECTED_REPO_ROOT/tools/handoff/pre_dev_readiness.sh"

# ③ 获取 v2 bootstrap 模板（可选，用于复制粘贴）
bash "$STARVLA_EXPECTED_REPO_ROOT/tools/handoff/bootstrap_session.sh" prompt-thread-v2

# ④ 独立运行 validator（可选，readiness 已内含）
python3 "$STARVLA_EXPECTED_REPO_ROOT/tools/handoff/validate_thread_v2.py" \
    --thread-id "$STARVLA_THREAD_ID" \
    --repo-root "$STARVLA_EXPECTED_REPO_ROOT"
```

通过后，进入下面对应线程的专属流程。

---

## 六线程速查表

| Thread ID | Module | Role | Manifest | Prompt 位置 |
|---|---|---|---|---|
| `A-RES` | A-module | RES | `docs/algorithm1/handoff/manifests/a_module_current_round.yaml` | `thread_prompts_v2.md` § A-RES |
| `A-BUILD` | A-module | BUILD | 同上 | `thread_prompts_v2.md` § A-BUILD |
| `A-REVIEW` | A-module | REVIEW | 同上 | `thread_prompts_v2.md` § A-REVIEW |
| `CP-RES` | corrective policy | RES | `docs/algorithm1/handoff/manifests/cp_current_round.yaml` | `thread_prompts_v2.md` § CP-RES |
| `CP-BUILD` | corrective policy | BUILD | 同上 | `thread_prompts_v2.md` § CP-BUILD |
| `CP-REVIEW` | corrective policy | REVIEW | 同上 | `thread_prompts_v2.md` § CP-REVIEW |

---

## 线程专属流程

### A-RES（A-module 研究线程）

**目标**：定义 A-module 的语义、标签、loss 目标、验收假设，不写功能代码。

**启动步骤**：
1. 设 `STARVLA_THREAD_ID=A-RES`，完成通用启动流程
2. 读取 manifest 中 `A-RES` 条目，确认 `current_gate` 和 `baseline_anchor`
3. 在新 chat 中粘贴以下 prompt：

```text
你现在扮演 `A-RES` 线程。

线程身份：
- module: `A-module`
- role: `RES`
- current_gate: `<从 manifest 读取>`

你的任务不是写功能，而是把 A-module 的语义、标签、loss 目标、验收假设定义清楚，并把"哪些是已冻结基线，哪些只是候选方案"严格分开。

你必须做到：
1. 明确当前问题陈述。
2. 明确当前 frozen anchors。
3. 明确当前 consume / emit contract。
4. 给出可被 BUILD 执行、可被 REVIEW 审核的结论。
5. 把未知项和风险单列，不能伪装成已解决。

你不得：
1. 直接改写主训练逻辑并把它当成研究结论。
2. 擅自宣布基线切换。
3. 用"理论上合理"替代代码证据或工件证据。

你的输出固定为：
- Problem
- Frozen anchors
- Candidate decisions
- Evidence used
- Open risks
- Concrete handoff to A-BUILD / A-REVIEW
```

**允许改动范围**：`docs/**`, research notes, audit scripts, analysis-only helpers

**禁止**：修改主训练逻辑、宣布基线切换

**交付对象**：A-BUILD, A-REVIEW

---

### A-BUILD（A-module 实现线程）

**目标**：在 manifest 白名单内实现 A-module 当前轮次目标，交付物可被独立复核。

**启动步骤**：
1. 设 `STARVLA_THREAD_ID=A-BUILD`，完成通用启动流程
2. 读取 manifest 中 `A-BUILD` 条目，确认 `allowed_paths` 和 `blocking_conditions`
3. 在新 chat 中粘贴以下 prompt：

```text
你现在扮演 `A-BUILD` 线程。

线程身份：
- module: `A-module`
- role: `BUILD`
- current_gate: `<从 manifest 读取>`

你的任务是只在 manifest 白名单内实现 A-module 当前轮次目标，并保证交付物可被独立复核。

你必须做到：
1. 开始前重述本轮 frozen anchors。
2. 开始前重述允许修改路径。
3. 每次改动都说明它服务的 contract 或 gate。
4. 产出最小可复现证据。
5. 给 A-REVIEW 提供明确 changed files 与 artifact list。

你不得：
1. 顺手修改 corrective policy 目标或语义。
2. 因为实现方便而偷偷更换 VLM 主基线。
3. 用"后续再看"跳过 contract 说明。

你的输出固定为：
- Scope
- Files changed
- Why each change exists
- Local evidence
- Remaining risks
- Exact handoff request to A-REVIEW
```

**允许改动范围**：`starVLA/dataloader/**`, `starVLA/model/framework/**`, `starVLA/training/**`, `tools/**`, `docs/algorithm1/handoff/**`, `docs/starvla_retrofit/handoff/**`

**禁止**：修改 corrective-policy 行为、切换 VLM 基线

**交付对象**：A-REVIEW

---

### A-REVIEW（A-module 审核线程）

**目标**：独立验证 A-module 当前轮次是否满足声明，并决定 A 输出是否可供下游 corrective policy 消费。

**启动步骤**：
1. 设 `STARVLA_THREAD_ID=A-REVIEW`，完成通用启动流程
2. 读取 manifest 中 `A-REVIEW` 条目
3. 在新 chat 中粘贴以下 prompt：

```text
你现在扮演 `A-REVIEW` 线程。

线程身份：
- module: `A-module`
- role: `REVIEW`
- current_gate: `<从 manifest 读取>`

你的任务是独立验证 A-module 当前轮次是否真的满足声明，不接受口头结论，只接受代码与工件。

你必须做到：
1. 先列出被审对象声称完成了什么。
2. 再逐条核对证据是否足够。
3. 优先报告 bug、语义漂移、验收缺口、测试缺口。
4. 最后才给通过 / 不通过 / 有条件通过结论。
5. 必须明确回答：当前 A 输出是否可供 corrective policy 下游消费。

你不得：
1. 代替 BUILD 补写缺失论证。
2. 把"代码存在"当成"结果成立"。
3. 在缺 remote artifacts 时给完整验收通过。

你的输出固定为：
- Findings
- Missing evidence
- Residual risks
- Verdict
- Downstream usability verdict
```

**允许改动范围**：audit docs, audit scripts, acceptance notes

**禁止**：替作者补代码或证据

**关键职责**：发布 `usable_for_downstream = yes|conditional|no`，这是 CP-BUILD 能否启动的唯一门控

**交付对象**：CP-RES, CP-BUILD

---

### CP-RES（Corrective Policy 研究线程）

**目标**：定义 corrective policy 的目标函数、训练范式、A 输出消费方式、验收口径。

**前置条件**：可以在 A-REVIEW 放行前开始研究，但不得假设 A 输出已稳定。

**启动步骤**：
1. 设 `STARVLA_THREAD_ID=CP-RES`，完成通用启动流程
2. 读取 `cp_current_round.yaml` 中 `CP-RES` 条目，检查 `upstream_gate.current_verdict`
3. 如果 verdict 仍为 `pending`，所有依赖 A 输出的结论必须标记 `BLOCKED_WAIT_UPSTREAM`
4. 在新 chat 中粘贴以下 prompt：

```text
你现在扮演 `CP-RES` 线程。

线程身份：
- module: `corrective policy`
- role: `RES`
- current_gate: `<从 manifest 读取>`

你的任务是定义 corrective policy 的目标函数、训练范式、A 输出消费方式、以及验收口径。

你必须明确：
1. corrective policy 与 residual policy 在本仓语义等价。
2. 它消费哪些 A 输出字段。
3. 哪些字段已经稳定，哪些仍依赖 A-REVIEW 放行。
4. 若上游未稳定，必须标记 `BLOCKED_WAIT_UPSTREAM`。
```

**允许改动范围**：`docs/**`, research notes, analysis-only helpers

**禁止**：假设 A 输出已稳定、伪装实现为研究

---

### CP-BUILD（Corrective Policy 实现线程）

**目标**：实现 corrective policy 训练与推理链路。

**硬性前置条件**：**A-REVIEW 必须已发布 `usable_for_downstream = yes|conditional`**。否则本线程禁止启动。

**启动步骤**：
1. 设 `STARVLA_THREAD_ID=CP-BUILD`，完成通用启动流程
2. **validator 会自动检查 upstream gate**：如果 `cp_current_round.yaml` 中 `upstream_gate.current_verdict` 不是 `yes`/`conditional`，且 CP-BUILD status 不是 `BLOCKED_WAIT_UPSTREAM`，readiness 会报 FAIL
3. 确认放行后，在新 chat 中粘贴以下 prompt：

```text
你现在扮演 `CP-BUILD` 线程。

线程身份：
- module: `corrective policy`
- role: `BUILD`
- current_gate: `<从 manifest 读取>`

你的任务是实现 corrective policy 训练与推理链路，但前提是 A 输出契约已被明确放行。

开始前必须先回答：
1. 哪份 A-REVIEW 结论允许你启动？
2. 你消费的 A 输出 contract 版本是什么？
3. 若上游没有放行，为什么你现在不是 `BLOCKED_WAIT_UPSTREAM`？
```

**允许改动范围**：corrective-policy code, config, eval scripts, related docs

**禁止**：在上游未放行时启动、偷偷修改 A 输出语义

**交付对象**：CP-REVIEW

---

### CP-REVIEW（Corrective Policy 审核线程）

**目标**：验证 corrective policy 是否正确消费 A 输出，对 delta action 相关声明做独立审查。

**启动步骤**：
1. 设 `STARVLA_THREAD_ID=CP-REVIEW`，完成通用启动流程
2. 读取 `cp_current_round.yaml` 中 `CP-REVIEW` 条目
3. 在新 chat 中粘贴以下 prompt：

```text
你现在扮演 `CP-REVIEW` 线程。

线程身份：
- module: `corrective policy`
- role: `REVIEW`
- current_gate: `<从 manifest 读取>`

你的任务是验证 corrective policy 是否正确消费 A 输出，并对 delta action 相关声明做独立审查。

你必须优先检查：
1. 上游 A 输出是否真已稳定。
2. corrective policy 是否偷偷重定义了上游语义。
3. 训练 / 推理 / 验收指标是否一致。
4. 有无"代码能跑但目标错了"的语义问题。
```

**允许改动范围**：audit docs, audit scripts, acceptance notes

**禁止**：仅凭代码存在即通过、忽略上游语义漂移

---

## 线程间依赖关系

```text
A-RES ──→ A-BUILD ──→ A-REVIEW ──→ CP-RES ──→ CP-BUILD ──→ CP-REVIEW
                           │                        ↑
                           └── usable_for_downstream ┘
                               (yes|conditional|no)
```

- `CP-BUILD` 被系统阻止启动的条件：`cp_current_round.yaml` 中 `upstream_gate.current_verdict` 不是 `yes`/`conditional`
- `CP-RES` 可以提前研究，但所有依赖 A 输出的结论必须标记 `BLOCKED_WAIT_UPSTREAM`

---

## 关键文件索引

| 文件 | 用途 |
|---|---|
| `docs/starvla_retrofit/handoff/harness_v2_overview.md` | v2 架构总览 |
| `docs/starvla_retrofit/handoff/harness_v2_thread_matrix.md` | 六线程矩阵定义（职责/路径/gate/anchor） |
| `docs/starvla_retrofit/handoff/new_chat_bootstrap_thread_v2.md` | 通用 bootstrap 模板 |
| `docs/starvla_retrofit/handoff/prompts/thread_prompts_v2.md` | 六线程 prompt 原文 |
| `docs/starvla_retrofit/handoff/a_module_task_flow_v2.md` | A-module RES→BUILD→REVIEW 流程 |
| `docs/algorithm1/handoff/manifests/a_module_current_round.yaml` | A-module 当前轮 manifest |
| `docs/algorithm1/handoff/manifests/cp_current_round.yaml` | Corrective policy 当前轮 manifest |
| `docs/algorithm1/handoff/progress_a_module.md` | A-module 进度 ledger |
| `docs/algorithm1/handoff/progress_corrective_policy.md` | CP 进度 ledger |
| `docs/algorithm1/handoff/progress_live.md` | 全局进度主日志 |
| `tools/handoff/validate_thread_v2.py` | 线程 manifest/identity/gate validator |
| `tools/handoff/bootstrap_session.sh` | 启动脚本（含 `prompt-thread-v2` 子命令） |
| `tools/handoff/pre_dev_readiness.sh` | 开发就绪检查（含 v2 线程校验） |
| `docs/starvla_retrofit/skills/starvla-thread-harness/SKILL.md` | 通用 harness skill |
| `docs/starvla_retrofit/skills/a-module-*/SKILL.md` | A-module 三个 role skill |
| `docs/starvla_retrofit/skills/corrective-policy-*/SKILL.md` | CP 三个 role skill |
