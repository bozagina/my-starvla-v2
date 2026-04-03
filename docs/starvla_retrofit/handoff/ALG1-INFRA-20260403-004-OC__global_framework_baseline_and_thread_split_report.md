# ALG1-INFRA-20260403-004-OC 全局框架基线审计与线程拆分报告

## 1) 当前状态与约束（<=8条）

1. 当前分支为 `codex/tmp-20260402-p0-audit-infra`，三层分支拓扑通过 (`tmp -> worktree -> final`)。
2. 工作区仍是 dirty（4 路径），本轮只新增/更新文档，不回滚现有改动。
3. Chunk 输出已具备: `MapAnythingLlava3DPI.predict_action` 返回 `normalized_actions`，并支持 `return_debug_info`。
4. Episode replay 能力存在于数据与评测两端: dataset 基于 episode/chunk 读取，eval 客户端按 chunk 缓存动作。
5. Windowed sampling 已有基础: `ModalityConfig.delta_indices` + `retrieve_data_and_pad`；但 steps cache 文件名固定，存在配置漂移风险。
6. Trainer 目前是单损失主路径（`action_loss`），`_train_step` 是 A/corrective 最小插入点。
7. 评估侧当前是“训练 batch 内评估 + mse_score”，缺少独立 held-out eval 通道。
8. 远程生效结论仍需用户提供远程日志/配置/指标，本地仅能给可执行性与接口结论。

## 2) 证据锚点（文件级）

1. `train_starvla.py`:
   - 训练主循环与 eval 调用: `train()` / `eval_action_model()`。
   - 插入点: `_train_step()`（loss 聚合、metrics 输出）。
2. `MapAnythingLlava3DPI.py`:
   - `future_action_window_size`/`chunk_len` 定义。
   - `predict_action(..., return_debug_info=...)` 输出 `normalized_actions`。
3. `datasets.py`:
   - `_get_all_steps()`、`sample_step()`、`delta_indices`、`retrieve_data_and_pad()`。
   - episode/chunk 读取路径 `get_trajectory_data*`。
4. `model2libero_interface.py` + `eval_libero.py`:
   - `action_chunk_size = future_action_window_size + 1`。
   - 逐步推理与成功率统计日志。

## 3) 最小可执行 P1 改造计划（按文件）

1. `/Users/bazinga/code/my-starvla-v2/starVLA/dataloader/gr00t_lerobot/datasets.py`
   - 增加共享 builder 输出结构（兼容旧字段）。
   - 加入最小 shape/字段校验与错误信息。
2. `/Users/bazinga/code/my-starvla-v2/starVLA/dataloader/lerobot_datasets.py`
   - 将 builder/schema 开关串到 dataset 构建入口。
3. `/Users/bazinga/code/my-starvla-v2/starVLA/training/train_starvla.py`
   - `_train_step` 增加可选 `a_loss/corrective_loss` 聚合插入点（默认关闭）。
4. `/Users/bazinga/code/my-starvla-v2/starVLA/model/framework/MapAnythingLlava3DPI.py`
   - 补齐可选 debug 输出与 trainer/eval 对齐字段。
5. `/Users/bazinga/code/my-starvla-v2/tools/handoff/*.py`
   - 增补 schema validator CLI（供本地与验收线程复用）。

## 4) 线程化交付

1. 已落地线程索引:
   - `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
2. 已落地 4 份线程专用文档:
   - T1 DATA
   - T2 SCHEMA/INFRA
   - T3 EVAL/DIAG/ACCEPTANCE
   - T4 TRAINER INSERTION

## 5) 结论

当前可以进入并行开发阶段，但必须按 gate 执行，并强制使用 `BLOCKED_WAIT_REMOTE` 机制避免验收/开发互相等待死锁。
