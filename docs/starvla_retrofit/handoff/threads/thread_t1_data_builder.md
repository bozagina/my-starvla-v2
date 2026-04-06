# T1(DATA): Shared Builder 与 Windowed Sampling

- 线程定位: `P1` 前置实现线程（先做接口与最小实现，不做训练大改）。
- 建议模块: `DATA`

## 1) 启动命令

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export STARVLA_EXPECTED_REPO_ROOT="<absolute-target-repo-root>"
export STARVLA_EXPECTED_VLM_SCOPE="qwen_only"
bash "$REPO_ROOT/tools/handoff/ensure_repo_context.sh" --expect-root "$STARVLA_EXPECTED_REPO_ROOT" --expect-vlm-scope "$STARVLA_EXPECTED_VLM_SCOPE" --require-expected-root
bash "$REPO_ROOT/tools/handoff/pre_dev_readiness.sh"
"$REPO_ROOT/tools/handoff/bootstrap_session.sh" start \
  --module DATA \
  --owner OC \
  --title "T1 shared builder + windowed sampling"
```

## 2) 必读上下文（按顺序）

1. `<REPO_ROOT>/docs/starvla_retrofit/handoff/context_pack_compact.md`
2. `<REPO_ROOT>/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
3. `<REPO_ROOT>/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
4. `<REPO_ROOT>/docs/starvla_retrofit/handoff/threads/thread_t1_data_builder.md`

## 3) 白名单文件（只在这些文件内改）

1. `<REPO_ROOT>/starVLA/dataloader/gr00t_lerobot/datasets.py`
2. `<REPO_ROOT>/starVLA/dataloader/lerobot_datasets.py`
3. `<REPO_ROOT>/starVLA/dataloader/__init__.py`
4. `<REPO_ROOT>/starVLA/dataloader/gr00t_lerobot/data_config.py`
5. `<REPO_ROOT>/starVLA/config/training/*.yaml`

## 4) 线程专用 Prompt（复制即用）

```text
你是 T1(DATA) 线程，目标是为 StarVLA retrofit 建立“共享 builder + 窗口抽样”最小可执行基础。

要求:
1) 先读取:
   - context_pack_compact.md
   - system_prompt_operating_contract.md
   - thread_prompts_and_checklists_index.md
   - thread_t1_data_builder.md
2) 仅在 dataloader 相关白名单文件改动。
3) 保持向后兼容: 旧 batch 字段不删除，新增字段走可选开关。
4) 产出:
   - builder 输入/输出字段定义
   - 最小实现与本地 smoke 证据
   - 提供给 T2/T4 的字段清单
5) 每轮输出必须包含:
   - 修改了什么
   - 证据是什么
   - 得到什么结论
   - 下一步建议
   - 建议 commit message
```

## 5) Upstream Pass Checks

1. `pre_dev_readiness.sh` 为 `READY_TO_DEVELOP=YES`。
2. 当前会话处于 `codex/tmp-*`。
3. `P0` 审计结论已确认: `delta_indices + retrieve_data_and_pad` 可支持窗口抽样。

## 6) Checklist

1. 明确 builder 最小 schema: `obs`, `action_chunk`, `lang`, `state(optional)`, `meta`。
2. 把窗口参数和 `future_action_window_size` 对齐，避免训练/评测 chunk 长度漂移。
3. 增加最小校验逻辑（shape/字段存在性），错误时可解释。
4. 输出兼容说明: 哪些旧字段保留，哪些新字段为可选。

## 7) Done 证据格式

1. 关键命令: `rg -n "delta_indices|retrieve_data_and_pad|sample_step|future_action_window_size" ...`
2. 本地 smoke: 至少一次构造 batch 并打印字段 shape（可不依赖远程）。
3. `progress_live.md` 记录“对 T2/T4 可消费字段清单”。

## 8) 交接规则

1. 给 T2: 样本 schema 草案（字段名、dtype、shape、必填/可选）。
2. 给 T4: trainer 可直接消费的 `action_chunk` 与辅助标注字段。
3. 若数据依赖缺失导致无法实测，状态改 `BLOCKED_WAIT_REMOTE`，列明缺失数据与 owner。
