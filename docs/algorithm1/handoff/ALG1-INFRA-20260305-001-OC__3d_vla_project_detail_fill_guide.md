# 3D VLA 项目详报填写与改进指南（main -> 当前）

- EXP_ID: `ALG1-INFRA-20260305-001-OC`
- 适用对象：组内协作成员、导师评审、后续接手模型/工程师
- 对应初稿模板：`/Users/bazinga/Downloads/3D VLA project（detail）.docx`
- 目标：把当前“PRD 样式初稿”升级为“可管理、可追溯、可执行”的项目主文档

---

## 0. 当前初稿的优点与缺口

### 已有优点

1. 已覆盖标准 PRD 主干（目标、方案、实验、里程碑、风险）。
2. 结构利于汇报，不需要重起炉灶。
3. 已有“多模态 + correction”主线，可快速对齐技术方向。

### 关键缺口（建议优先补齐）

1. **缺少版本演进主线**：从 `main baseline` 到 `Path-A v4-1-1` 的因果链还不够明确。
2. **缺少证据绑定**：多数结论未绑定 EXP_ID / run_id / 指标截图。
3. **缺少管理可执行性**：里程碑未绑定 Owner、截止时间、Exit Criteria。
4. **缺少本地-远程边界说明**：当前项目是“本地开发、远程训练”，文档里应显式写清。
5. **缺少决策记录层**：哪些假设已证伪（如 tower 主因）应固化，避免团队反复争论。

---

## 1. 建议的文档定位（两层结构）

建议把这份文档定位为“管理主文档”，并配套“技术证据附录”：

- 主文档（面向导师/协作）
  - 决策、范围、里程碑、风险、任务分工。
- 技术附录（面向实现/复现）
  - 代码锚点、实验编号、核心指标曲线与日志链接。

建议在主文档开头增加一行：

> 本文是项目管理主文档，所有技术细节和证据索引以附录与 handoff 文档为准。

---

## 2. 来龙去脉（可直接粘贴到 B/C 节）

> 从 `main` 分支 baseline 开始，系统最初采用“语言特征 + 视觉特征 + 几何特征直接拼接”作为动作主干输入，训练稳定但缺少显式时序纠偏信号。随后引入 Path-A：早期 `v3/v4` 使用 pooled `delta_z` 做反馈注入，证明可运行但增益偏弱；演进到 `v4-1` 后引入有界 delta-action（`a_out=a_base+gate*clip(DeltaA)`）提升了稳定性与可控性；当前主线 `v4-1-1` 进一步升级为 token-level `Δgeo + soft-mask + teacher`。在该阶段，已证伪“视觉 tower 来源错配是主因”（current/llava_rebuild/official 的 alpha cosine 均约 `0.995+`），并定位到核心矛盾是 mask 监督可辨识性不足与实现尺度压平问题。EXP003 中 per-head L2 修复后，`entropy/topk/alpha_max/logits_std` 出现持续改善，验证了当前路线有效；后续重点转向降低 delta-action 后段饱和并提升 teacher 分离度。

---

## 3. 逐节填写指南（对应你模板 A-L）

## A. Project Card（项目卡片）

建议新增字段：

- `Project`: 3D VLA / Path-A causal feedback
- `Current Version`: `v4-1-1`
- `Current Phase`: supervision-strengthening + stability optimization
- `Latest EXP`: `ALG1-INFRA-20260305-001-OC`（文档管理）
- `Core Code`: `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`
- `Tracking Doc`: `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md`

## B. Executive Summary（10 行内）

建议用“四句法”：

1. 问题：2D/弱几何先验在 3D 操作中存在空间歧义与纠偏不足。
2. 路线：从拼接 baseline 迭代到 Path-A `v4-1-1`（token-level feedback）。
3. 证据：已证伪 tower 主因；per-head L2 修复后 mask 选择性显著提升。
4. 下一步：控制 clip 饱和 + teacher 分离度 + 批量诊断闭环。

## C. 背景与目标（Goals / Non-goals）

### C1 背景建议填写

- 场景：`LIBERO + 远程训练 + 本地分析`。
- 输入：语言、multi-view RGB、几何 token。
- 输出：动作 chunk（当前常见 `future_action_window_size+1`）。

### C2 Goals 建议拆成双目标

- G1（性能）：`action_dit_loss` 不退化，目标提升任务成功率。
- G2（可解释）：`soft_mask` 从高熵近均匀向更稀疏/可区分迁移。

### C3 Non-goals（建议写死，防止范围膨胀）

- 本阶段不做全新 VLM 大迁移（如直接替换到 Qwen2.5-VL 生产化落地）。
- 本阶段不做全套 world model 闭环。

## D. Usage Scenarios

建议写 2 个场景即可：

1. 训练期：`t/t+k` 双时刻构造残差，指导纠偏学习。
2. 推理期：维持 history/chunk，按控制模式执行 correction（chunk / receding_horizon）。

## E. Scope & Deliverables

### In Scope（建议本轮）

1. mask 监督增强（teacher + contrast + 对齐诊断）。
2. delta-action 饱和治理（最小 CFG A/B）。
3. 批量诊断与自动化报表（而非单样本结论）。

### Out of Scope（建议）

1. 全系统迁移到新 VLM 主干。
2. 多本体泛化 benchmark 全覆盖。

### Deliverables（建议固定四件）

1. 代码改动（含 config/run_id 可追溯）。
2. 远程 run 证据包（config/metrics/log/run_identity）。
3. 对照报告（baseline vs ablation）。
4. 面向管理的一页摘要。

## F. 方案设计（Design）

建议把 F2 模块改成“三模块固定口径”（与你现有积累一致）：

1. **M1 表征构建**：`z_t`（稳定，非主瓶颈）
2. **M2 残差+mask**：`Δgeo + alpha`（当前主战场）
3. **M3 注入利用**：delta-action（有效但后段饱和）

每个模块统一 6 行：

- 输入
- 输出
- 关键公式
- 代码锚点（绝对路径）
- 当前状态（Done/In-progress）
- 风险与 fallback

## G. 数据与训练计划

建议补 3 个管理关键项：

1. `run_id` 必须带 `EXP_ID`（用于远程映射）。
2. 每次训练分析必须附 `run_identity.txt`。
3. 明确“本地改动不等于远程生效”。

可直接引用流程文档：
`/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/remote_training_workflow.md`

## H. 实验与评测

建议把你模板 H2/H3 的抽象项改成“当前已执行 + 下一步待执行”双列表。

### 已执行且有证据

1. tower 对照诊断（current/llava/official）：主因被证伪。
2. directed_self_cross 对照：可用但非单点突破。
3. teacher 门控修正：`confidence_floor 0.02 -> 0.002` 后 KL 生效。
4. per-head L2 修复：选择性指标显著改善并可持续。

### 下一步必做

1. delta-action 饱和 A/B（核心瓶颈验证）。
2. token mismatch 数据集覆盖验证。
3. teacher temperature/smoothing 校准。

## I. 里程碑与执行计划

建议你现在就填成“3 个 2 周里程碑”：

- M1（T+2周）：批量诊断上线 + 指标看板固定化。
- M2（T+4周）：clip saturation 降低并验证无主损回归。
- M3（T+6周）：形成论文/汇报级证据包（图表+失败案例+结论）。

每个里程碑必须写：`Owner / ETA / Exit Criteria / 证据路径`。

## J. 当前进展与证据

建议改为“三栏固定模板”：

1. `Done`：按 EXP_ID 列（结论 + 链接）
2. `Open Issues`：现象 + 复现条件 + 影响
3. `Next Week`：3 条以内，均可验收

可优先引用：

- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/mask_diagnosis_full_history.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260302-001-OC__path_a_end_to_end_guide.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md`

## K. 风险与对策

建议用“触发信号 -> 应对 -> 回退”的格式：

1. **clip saturation 高** -> 调整 delta-action clip/alpha schedule -> 必要时回退到稳态配置。
2. **teacher 分离度低** -> 调整 temperature/smoothing/floor -> 必要时降权并保留日志监控。
3. **远程-本地版本错配** -> 强制 run_id+EXP_ID 映射 -> 无映射 run 标记弱可追溯。

## L. 附录（复现与引用）

建议新增两项：

- `Decision Log (ADR)`：记录“选了什么，不选什么，为什么”。
- `Changelog`：按日期追加（便于组内同步）。

---

## 4. 建议加入的“证据最小集”

每次汇报最少应附：

1. 训练配置：`config.yaml`
2. 指标：`metrics.jsonl` + `summary.jsonl`
3. 日志：`train.log` / `train.raw.log`
4. 身份映射：`run_identity.txt`
5. 进度记录：`progress_live` 对应条目

---

## 5. 你这版文档建议立即改的 8 处（最小改动优先）

1. 在 B 节加入“main -> v3/v4 -> v4-1 -> v4-1-1”一段话。
2. 在 C 节增加 Non-goals，防止范围失控。
3. 在 F2 统一为 M1/M2/M3 三模块口径。
4. 在 G 节补“本地开发-远程训练边界”。
5. 在 H 节把“抽象 ablation”换成“已做/待做”两组。
6. 在 I 节每个里程碑补 Owner+Exit Criteria。
7. 在 J 节所有 Done 条目补 EXP_ID + 证据链接。
8. 在末尾新增 ADR + Changelog 两个区域。

---

## 6. 可直接粘贴的“本周 Next”示例

- 完成 `delta_action_clip_saturation_frac` 降饱和最小 CFG A/B，并汇总 2k/5k/10k 指标窗口。
- 在触发 token mismatch 的样本上验证 range-resample 对选择性与稳定性的影响。
- 产出一版导师汇报用 1 页版（管理）+ 1 版技术附录（证据）。

