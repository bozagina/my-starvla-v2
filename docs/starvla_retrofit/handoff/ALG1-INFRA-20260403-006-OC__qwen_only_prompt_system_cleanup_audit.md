# ALG1-INFRA-20260403-006-OC Qwen-only Prompt/System 清理审计

- 审计日期: 2026-04-03
- 审计目标: 将当前 StarVLA retrofit 活跃入口统一为 `Qwen2.5VL/Qwen3VL`，移除 `MapAnything/LLaVA3D` 作为当前主线依赖。

## 1) 当前状态与约束（<=8条）

1. 当前分支仍为 `codex/tmp-20260402-p0-audit-infra`，满足 `tmp -> worktree -> final` 的三层流转要求。
2. 本轮仅做 prompt/system/docs/script 入口清理，不做训练框架大重构。
3. 活跃入口文档已经补充硬约束：本仓当前仅走 `Qwen2.5VL / Qwen3VL`。
4. T4 线程白名单已从 `MapAnythingLlava3DPI.py` 切换到 Qwen framework + Qwen VLM module。
5. 远程工具默认路径已切到 `/2025233147/zzq_0317/starVLA`，并保留环境变量覆盖能力。
6. 历史审计报告与旧进度记录保留为历史证据，不作为新线程默认入口。
7. 远程是否生效仍需用户提供远程日志/配置/指标，本地只能验证入口与流程一致性。
8. `P0 gate` 仍未跳过；当前属于“先清理入口、再进入实现”的门控准备动作。

## 2) 证据锚点

1. 活跃文档清理:
   - `docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
   - `docs/starvla_retrofit/handoff/context_pack_compact.md`
   - `docs/starvla_retrofit/handoff/new_chat_bootstrap_compact.md`
   - `docs/starvla_retrofit/handoff/retrofit_prompt_and_branch_strategy.md`
   - `docs/starvla_retrofit/handoff/development_run_checklist.md`
   - `docs/starvla_retrofit/handoff/v2_quickstart.md`
   - `docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
   - `docs/starvla_retrofit/handoff/threads/thread_t4_trainer_insertion.md`
2. 任务书范围声明更新:
   - `docs/starvla_retrofit/handoff/star_vla改造与统一伪标签生成任务书.md`
3. 远程默认路径更新:
   - `tools/fetch_latest_run_files.sh`
   - `tools/run_remote_train.sh`
   - `tools/deploy_to_server.sh`
   - `tools/tail_latest_trainlog.sh`
4. 历史保留（未改）:
   - `docs/algorithm1/handoff/**` 中大量 `MapAnything/LLaVA3D` 记录属于历史阶段，不作为当前 prompt/system 默认入口。

## 3) 最小可执行 P1 改造计划（按文件）

1. `/Users/bazinga/code/my-starvla-v2/starVLA/training/train_starvla.py`
   - 抽象当前 `cfg.framework.mapanything_llava3d.base_vlm` 读取点，统一为 Qwen 路径可读字段（保留兼容 alias）。
2. `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/__init__.py`
   - 固化 Qwen framework 选择逻辑与错误提示，避免回退到 MapAnything 依赖。
3. `/Users/bazinga/code/my-starvla-v2/starVLA/model/modules/vlm/__init__.py`
   - 统一 `qwen2.5/qwen3` 的路径匹配与构建分支。
4. `/Users/bazinga/code/my-starvla-v2/starVLA/config/training/*.yaml`
   - 增加/统一 Qwen-only 示例配置，并标记旧 `*_mapanything_llava3d.yaml` 为 legacy。
5. `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
   - 持续维护线程白名单与 Qwen-only 限制，作为并行开发的入口真相。

## 4) 结论

当前已具备“Qwen-only 入口一致性”，可以在不引入 3D VLM 依赖的前提下继续后续 P0/P1 开发。下一步应进入代码层的最小兼容改造（优先 `train_starvla.py` 的配置字段抽象）。

建议 commit message:
- `[ALG1-INFRA-20260403-006-OC] enforce qwen-only prompts docs and remote defaults`
