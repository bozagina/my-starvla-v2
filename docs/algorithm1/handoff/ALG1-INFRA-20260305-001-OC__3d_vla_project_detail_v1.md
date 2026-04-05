# 3D VLA 项目详报 v1（main -> 当前）

- EXP_ID: `ALG1-INFRA-20260305-001-OC`
- 文档定位：项目管理主文档（决策/范围/计划）+ 技术证据索引
- 更新时间：2026-03-06
- 适用范围：组内共享、导师汇报、后续接手执行

> 说明：本项目采用“本地开发、远程训练”。本文对训练效果的判断仅依据已回传的远程 `config/metrics/log/run_identity` 证据。

---

## 0. 团队与分工（展示区，待补充）

> 本节用于组会/导师汇报首页展示，强调“谁在做什么、当前有几条工作分支”。  
> 口径说明：这里的“工作分支”指任务主线/子线，不等同于 Git 分支。

### 0.1 人员与工作分配（请补充）

| 人员 | 角色 | 负责模块 | 当前任务 | 状态 | 本周交付 |
|---|---|---|---|---|---|
| `<姓名A>` | `<负责人/算法>` | `<P0主线验证>` | `<如：M00/M10结果闭环>` | `<IN_PROGRESS>` | `<表格/结论>` |
| `<姓名B>` | `<训练/平台>` | `<DIAG/INFRA>` | `<如：run证据包整理>` | `<IN_PROGRESS>` | `<脚本/报告>` |
| `<姓名C>` | `<评测/分析>` | `<LIBERO评测>` | `<如：5k/10k/20k对照>` | `<IN_PROGRESS>` | `<图表/摘要>` |
| `<姓名D>` | `<机制预研>` | `<Path-A预研（P1）>` | `<仅维护，不扩展>` | `<PARKED>` | `<风险记录>` |

### 0.2 当前工作分支总览（用于展示）

| 分支ID | 分支名称 | 目标 | 当前状态 | 优先级 | 进入/退出条件 |
|---|---|---|---|---|---|
| W1 | 阶段1主验证（2D/3D×geo） | 回答“3D VLM + action 是否提升” | `ACTIVE` | `P0` | 完成 M00/M01/M10/M11 可比证据 |
| W2 | 阶段2-A（3D基座机制放大） | 若3D成立，放大3D优势 | `PARKED` | `P1` | Gate-A/B 通过后进入 |
| W3 | 阶段2-B（2D基座+Path-A） | 若3D劣势，切换2D主线继续机制开发 | `PARKED_CANDIDATE` | `P1` | `M10<M00` 且 `M11<M01` 触发 |
| W4 | 支撑线（DIAG/INFRA） | 证据治理、追溯、评测规范 | `ACTIVE` | `P0` | 持续服务W1判定闭环 |

汇报建议一句话：

- 当前共 `4` 条工作分支，其中 `P0 ACTIVE=2`（W1/W4），`P1 PARKED=2`（W2/W3）；先完成 W1 再做阶段2路线选择。

---

## A. 项目卡片（Project Card）

- 项目名称：3D VLA Path-A 因果反馈增强
- 当前主线版本：`v4-1-1`
- 项目阶段：阶段1价值验证 + 阶段2机制增强（当前处于阶段1优先闭环）
- 主代码入口：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`
- 进度日志：`/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md`
- 核心总目标：构建可稳定落地的 3D VLA（3D 感知 + 动作生成）
- 一级任务（优先级最高）：验证 `main` 分支“3D VLM + action model”是否能稳定提升效果，并量化提升幅度
- 二级任务（条件触发）：当“单纯3D VLM”提升不足时，探索几何先验注入与残差机制以放大优势
- 当前执行优先级：`P0=main分支验证闭环`，`P1=Path-A机制增强（暂缓新增扩展）`
- 当前主风险：阶段1证据闭环不足（可比矩阵结果不完整）

---

## B. 决策摘要（Executive Summary）

1. 总目标：构建可落地的 3D VLA。  
2. `P0`（当前最高优先）：先完成 `main` 分支验证，回答“3D VLM + action 是否优于2D基线”。  
3. 验证口径：统一预算下完成 `M00/M01/M10/M11`，再做 Gate 判定。  
4. 硬规则：若 `M10<M00` 且 `M11<M01`，则切换到 `2D VLM + Path-A` 主线。  
5. `P1`（条件触发）按模块推进：  
   - 模块1：如何获取高质量几何特征；  
   - 模块2：如何构建高质量几何残差；  
   - 模块3：几何残差如何有效影响动作生成。  
6. 当前状态：阶段1结果闭环中；10k 无 run_id 结果仅作预观察，不用于最终分叉决策。

---

## C. 背景与目标（Goals / Non-goals）

### C1. 背景（Context）

- 场景：桌面操作任务（LIBERO 为主，真机为后续高优先验证）。
- 输入：语言指令 + 多视角图像 + 几何特征。
- 输出：动作 chunk（常见 `future_action_window_size + 1`）。
- 现象->后果：
  - 仅依赖2D先验 -> 深度歧义/遮挡导致抓取偏差与接触失败。
  - 单纯“上3D模型”不一定带来动作收益，需先做可比验证再决定后续路线。

### C2. 两阶段目标（按优先级执行）

#### 阶段1（P0，必须先完成）：主线价值验证

- 核心问题：`main` 分支中 `3D VLM + action model` 是否真的优于 2D 基线？
- 验证矩阵：`M00/M01/M10/M11`（2D/3D × geo off/on）。
- 统一口径：
  - 主指标：`Success@LIBERO`
  - 训练代理：`action_dit_loss`
  - 工程约束：`p95 latency`、稳定性（nonfinite/崩溃）

#### 阶段2（条件触发）：机制开发与放大

- 分叉 A（3D 优势成立）：继续以 3D VLM 为基座，探索机制放大（例如 Path-A）。
- 分叉 B（3D 全面劣势）：切换到 2D VLM 基座，再进行 Path-A 等机制开发。
- `P1` 模块化拆分（统一口径）：
  - 模块1：高质量几何特征获取；
  - 模块2：高质量几何残差构建；
  - 模块3：残差到动作生成的有效传导。

### C3. 关键决策规则（硬约束）

- 若阶段1验证满足 `M10<M00` 且 `M11<M01`，则认定“3D（含/不含geo）均劣于2D”：
  - 立即切换到 `2D VLM + Path-A` 主线；
  - 3D 路线降级为储备方向，不占用当前主资源。

### C4. Non-goals（当前阶段明确不做）

- 在阶段1结论未闭环前，不新增高成本 Path-A 扩展实验。
- 不做全栈 VLM 大迁移作为当前主线。
- 不做与阶段1主问题无关的大规模重构。

---

## D. 用户故事 / 使用方式（Usage Scenarios）

### 场景1：阶段1主验证（当前唯一主流程）

1. 以统一预算运行 `M00/M01/M10/M11`；
2. 回传每格标准证据（`run_identity/config/metrics/log`）；
3. 生成横向对照表，回答“3D是否提升、提升多少、代价多少”；
4. 根据门槛触发阶段2分叉（3D主线继续 / 2D基座切换）。

### 场景2：阶段2-A（仅当3D优势成立）

1. 保持 3D VLM 主线；
2. 引入机制增强（Path-A、残差利用、监督增强）；
3. 目标是放大已验证优势，而非重新证明基础价值。

### 场景3：阶段2-B（当3D全面劣于2D）

1. 主线切换到 2D VLM；
2. 在 2D 基座上继续 Path-A 机制研发；
3. 以“机制收益”而非“3D先验收益”为核心评估目标。

---

## E. 范围与交付物（Scope & Deliverables）

### E1. In Scope（当前）

1. `P0 主线验证`：完成阶段1四象限矩阵证据闭环。
2. `DIAG`：统一输出每个格子的结果摘要与门槛判定。
3. `INFRA`：确保证据可追溯（EXP_ID/run_id 对齐）。
4. `决策输出`：明确阶段2进入分叉A（3D继续）还是分叉B（切2D+Path-A）。

### E2. Out of Scope（当前）

1. 阶段1未完成前的新 Path-A 扩展。
2. 与阶段1决策无关的复杂改造。
3. 全量真机长时程验证（留到阶段2稳定后）。

### E3. 交付物

1. 阶段1矩阵结果总表（M00/M01/M10/M11）。
2. 每格标准证据包（`config/metrics/summary/log/run_identity`）。
3. 一页决策结论（继续3D 或 切换2D+Path-A）。
4. 更新后的管理主文档（含里程碑与资源重排建议）。

---

## F. 方案设计（Design）

### F1. 两阶段架构总览（先验证，再分叉）

- 阶段1：统一框架下做 2D/3D 与 geo on/off 可比验证。
- 阶段2-A：若3D优势成立，沿3D主线做机制增强。
- 阶段2-B：若3D全面劣势，切换到2D基座做Path-A增强。

### F2. 阶段1架构（当前执行架构）

#### V1 编码主干（可切换2D/3D）

- 输入：语言、图像（可选几何特征）。
- 输出：供 action model 使用的多模态表示。
- 要求：2D与3D除主干差异外，尽量保持训练与评测口径一致。

#### V2 动作模型（固定对照）

- 输入：VLM输出特征。
- 输出：动作 chunk。
- 要求：作为阶段1固定后端，避免“同时改两处”影响归因。

#### V3 评测与判定层

- 输入：四象限 run 结果。
- 输出：Gate-A/Gate-B 判定与阶段2分叉决策。
- 要求：结论可复现、可追溯、可审计。

### F3. 阶段2分叉架构（条件触发）

- 分叉A（3D继续）与分叉B（2D切换）均按同一 `P1 模块1~3` 推进：
  - 模块1（特征）：提高几何特征质量与稳定性；
  - 模块2（残差）：构建高质量几何残差并保证可辨识；
  - 模块3（作用）：验证残差对动作生成的实质影响与收益。
- 共同原则：机制开发服务于“提升动作效果”，不替代阶段1基础结论。

### F4. 关键接口与证据契约

- 训练接口：`run_id` 必须绑定 `EXP_ID`。
- 证据接口：每个run必须产出 `run_identity/config/metrics/summary/log`。
- 决策接口：矩阵结果必须可直接映射到 Gate 判定。

### F5. 时序与资源预算（按阶段）

- 阶段1：资源优先给四象限矩阵与复现实验。
- 阶段2：仅在阶段1结论明确后释放资源；分叉A/B二选一主推进。

---

## G. 数据与训练计划（Data / Training）

### G1. 数据与对齐

- 数据来源：示教数据 + 远程训练回传日志。
- 对齐重点：`t/t+k` 时序窗口、token 对齐、动作窗口一致性。
- 质量策略：记录 mismatch、coverage、teacher 可靠性指标。

### G2. 训练配置治理

- 每次有效实验必须有唯一 `EXP_ID`。
- 若改 YAML，`run_id` 必须带 `EXP_ID`（用于远程映射）。
- 无 `EXP_ID` 映射的 run，标记为弱可追溯，不做强结论。

### G3. 本地-远程边界（强制）

- 本地改动不等于远程生效。
- 远程是否生效以回传工件为准。
- 分析输入最低要求：`config + metrics + log + run_identity`。

---

## H. 实验与评测（Evaluation Plan）

### H1. 阶段1主验证：3D VLM + Action 是否带来提升（第一优先）

| ID | VLM | Geometric | 定位 |
|---|---|---|---|
| M00 | 2D | Off | 2D 基线下界 |
| M01 | 2D | On | 检验几何注入本身价值 |
| M10 | 3D | Off | `main` 分支核心对照（第一优先） |
| M11 | 3D | On | 3D + 几何联合上界 |

> 说明：阶段1先回答“3D VLM + action 本身是否有效”；阶段2才进入机制增强。

历史预结果纳入规则（新增）：

- 当前已有“3D with/without geometric”的 `10k checkpoint` 初步结果，但缺少 `run_id`。
- 这些结果应纳入报告作为 **预观察证据**，用于说明趋势，不用于最终路线判定。
- 在 `20k` 完整结果（含标准证据包）回传前，不触发 Gate-A/Gate-B/硬回退决策。

### H2. 阶段2机制实验（在阶段1不足背景下展开）

- B0：Path-A 早期版本（v3/v4 pooled residual）。
- B1：v4-1（有界 delta-action）。
- B2：v4-1-1（token-level residual + soft-mask + teacher）。
- B3：teacher 与对齐诊断增强版本（EXP002/EXP003 之后）。

> 当前优先级说明：本轮先不新增 Path-A 方向扩展；阶段2内容作为历史证据与候选后续路线保留。

### H3. 已执行关键消融 / 诊断（阶段2）

1. tower 对照诊断（current / llava rebuild / official）
   - 结论：alpha 差异极小，tower 非主因。
2. directed_self_cross 对照
   - 结论：路径可用，但非单点突破。
3. teacher 门控修正（`confidence_floor: 0.02 -> 0.002`）
   - 结论：teacher 从 KL=0 变为非零监督。
4. per-head L2 修复
   - 结论：选择性指标显著提升且长窗口保持。

### H4. 当前固定评测指标

- mask：`soft_mask_entropy`、`soft_mask_topk_mass_32`、`soft_mask_alpha_max_mean`、`soft_mask_attn_top1_mean_vis`
- feedback：`feedback_mask_contrast_term`、`feedback_mask_contrast_weighted`
- delta-action：`delta_action_alpha`、`delta_action_effective_norm_mean`、`delta_action_clip_saturation_frac`
- teacher：`soft_mask_teacher_kl`、`soft_mask_teacher_confidence_mean`、`soft_mask_teacher_neg_top1_gap`

### H5. 下一轮必做实验（按主次）

1. 先补齐阶段1矩阵证据闭环（M00/M01/M10/M11 的可比输出）。
2. 固化阶段1统一汇总模板（每格 run_id + 指标 + 结论 + 风险）。
3. 对阶段1结论做一次复现实验（至少1个关键格子二次验证）。
4. 阶段2（Path-A）保持观察，不做新增扩展，待阶段1结论稳定后再恢复。

### H6. 决策树与验证矩阵（对齐项目图）

为避免“只看单次结果”导致路线摇摆，本项目采用阶段化决策树：

1. `S0` 定义统一预算与对照口径（2D/3D、geo on/off）。
2. `S1` 运行四象限最小矩阵（M00/M01/M10/M11）。
3. `S2` 按门槛判定 `3D > 2D` 或回退分支。
4. `S3/S4` 在优势分支上深挖几何残差机制与落地方式。
5. `S5` 固化主线并输出里程碑交付。

四象限矩阵（20k steps 起步）：

| ID | VLM | Geometric | 目标 |
|---|---|---|---|
| M00 | 2D | Off | 基线下界 |
| M01 | 2D | On | 检验几何注入本身价值 |
| M10 | 3D | Off | 检验3D先验价值 |
| M11 | 3D | On | 联合上界 |

关键分叉门槛（建议执行）：

- Gate-A（3D是否优于2D）：`Success@LIBERO` 提升 `>= +3pp`，且 `p95 latency` 不劣化超过 `20ms`。
- Gate-B（几何是否有效）：`with_geo - without_geo >= +2pp`，并伴随 mask/残差指标方向一致改善。
- Gate-C（几何残差是否可落地）：FM条件 / 实时纠偏 / RL状态三方向至少一个稳定获益。

硬回退规则（新增，优先级最高）：

- 若阶段1结论满足“3D VLM 在 `geo off` 与 `geo on` 两条线上均劣于2D对应基线”，则直接触发 **战略回退**：
  - 主线从“3D VLM优先”切换为“2D VLM优先”；
  - Path-A 后续开发在2D基座上继续推进；
  - 3D方向降为长期储备课题，不再占用当前主资源。

图示规范与节点样式见：`/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260305-001-OC__project_diagram_v2_spec.md`

---

## I. 里程碑与执行计划（Milestones）

### M1（截至 2026-03-19）诊断标准化

- 输出：阶段1四象限矩阵结果表（M00/M01/M10/M11）+ 统一证据包索引。
- Exit Criteria：四格均有可追溯 run 证据，且可直接横向比较。
- Owner：`INFRA + DIAG`。

### M2（截至 2026-04-02）增益稳态化

- 输出：阶段1关键结论复核（至少一格复现实验）+ 门槛判定（Gate-A/Gate-B）。
- Exit Criteria：可明确回答“3D VLM+action 是否提升、提升多少、在何条件下提升”。
- Owner：`INFRA + DIAG + CFG`。

### M3（截至 2026-04-16）汇报与论文素材化

- 输出：阶段1决策报告（继续3D主线 / 进入阶段2机制增强）+ 汇报图表包。
- Exit Criteria：团队对下一阶段主线达成一致，决策有证据支撑。
- Owner：`INFRA + 全模块协作`。

---

## J. 当前进展与证据（Status & Evidence）

### J0. 阶段进展总览（体现主次）

| 阶段 | 目标 | 当前状态 | 说明 |
|---|---|---|---|
| 阶段1 | 验证 `main` 分支 3D VLM + action 是否优于对照 | IN_PROGRESS（P0） | 当前最优先，需补齐四象限结果闭环 |
| 阶段2 | 在“单纯3D提升不足”下做机制增强 | PARKED（P1） | 保留已有结论，暂不新增扩展内容 |

阶段切换条件（必须执行）：

- 若阶段1验证结果为“3D（含/不含geo）均劣于2D”，则阶段2恢复时默认采用 **2D VLM + Path-A** 路线，而非继续加码3D主线。

### J1. Done（阶段2历史关键结论，当前用于参考）

1. `ALG1-MASK-20260301-002-OC`
   - 完成 teacher 路径接线与门控修正，teacher KL 非零生效。
2. `ALG1-MASK-20260301-003-OC`
   - 完成 per-head L2 修复并验证选择性显著提升。
3. tower 诊断链路（详见总诊断文档）
   - 证伪 tower 错配为主因，问题收敛到监督与实现细节。
4. `ALG1-INFRA-20260302-003-OC`
   - 完成 eval stack 审计并新增 receding-horizon 诊断能力。

### J2. Open Issues（可复现）

1. 阶段1矩阵的可比证据包仍需补全（当前报告以阶段2证据为主）。
2. `delta_action_clip_saturation_frac` 后段偏高。
3. teacher 分离度指标（如 `neg_top1_gap`）仍偏弱。
4. mismatch 分支触发样本不足，泛化收益证据不够。
5. 历史10k结果无 `run_id`，当前仅可作为弱可追溯预观察。

### J3. Next（本周）

1. 优先补齐阶段1矩阵对照结论（回答“3D VLM + action 是否提升、提升多少”）。
2. 统一整理四格实验的 run identity 与关键指标摘要（可直接贴报告）。
3. 完成一次阶段1关键结论复现实验，确保可重复。
4. 暂不新增 Path-A 扩展补充，待阶段1结论稳定后再评估恢复时点。

阶段1执行清单见：`/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260305-001-OC__main_branch_validation_checklist.md`

---

## K. 风险与对策（Risks）

1. 风险：clip 饱和导致纠偏幅度上限过早触顶。  
   - 触发信号：`delta_action_clip_saturation_frac` 持续高位。  
   - 对策：下调幅度/重设 schedule；必要时回退稳定配置。

2. 风险：teacher 监督存在“可用但分离弱”。  
   - 触发信号：`confidence_mean` 与 `neg_top1_gap` 低位。  
   - 对策：temperature/smoothing/floor 小步校准并保留负控验证。

3. 风险：远程 run 与本地改动映射错误。  
   - 触发信号：缺少 `run_identity` 或 `run_id` 无 `EXP_ID`。  
   - 对策：强制 run identity 检查；无映射 run 不用于最终结论。

---

## L. 附录（Appendix）

### L1. 关键参考文档

- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/mask_diagnosis_full_history.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260302-001-OC__path_a_end_to_end_guide.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260302-001-OC__report.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260305-001-OC__project_diagram_v2_spec.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260305-001-OC__main_branch_validation_checklist.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/remote_training_workflow.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/experiment_id_and_naming_convention.md`
- `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md`

### L2. 证据最小集（每次汇报必须）

1. `config.yaml`
2. `metrics.jsonl`
3. `summary.jsonl`
4. `train.log`（或 `train.raw.log`）
5. `run_identity.txt`

---

## M. 决策记录（ADR）

### ADR-001（已定）

- 决策：不把“更换 tower 来源”作为当前主攻方向。
- 原因：对照诊断中 alpha 差异极小，非主矛盾。
- 替代方案：转向监督增强与实现细节优化。

### ADR-002（已定）

- 决策：保留 per-head L2 修复为当前默认实现。
- 原因：已在远程 run 中体现持续选择性提升。
- 约束：继续监控主损与稳定性，防止副作用。

### ADR-003（待评审）

- 决策候选：是否提升 feedback contrast 权重并同步调整 teacher 温度。
- 评审条件：先完成阶段1主线验证结论，确认阶段2恢复优先级。

### ADR-004（新增待评审，决策优先级高）

- 决策候选：若阶段1确认3D全面劣于2D，是否正式切换到“2D VLM + Path-A”主线。
- 触发条件：`M10 < M00` 且 `M11 < M01`（在统一预算与可比条件下成立）。
- 执行动作：冻结3D主线扩展，迁移Path-A实验到2D基座并重建对照矩阵。

---

## N. 变更记录（Changelog）

- `2026-03-05`：创建本版详报 v1，完成从 `main` 到当前的统一管理口径。  
- `2026-03-06`：按项目图重构主次叙事，明确“一级任务=main分支价值验证；二级任务=提升不足时机制增强”，并将该逻辑写入评测与进展章节。  
- `2026-03-06`：进一步上调 `main` 分支验证优先级（P0），将 Path-A 调整为 P1 暂缓扩展，仅保留历史证据引用。  
- `2026-03-06`：新增硬回退规则：若阶段1显示3D（含/不含geo）均劣于2D，则切换到2D VLM基座继续Path-A开发。  
- `2026-03-06`：新增“团队与分工 + 工作分支总览”前置展示区，便于统一对外口径。  
- `2026-03-06`：简化决策摘要，并将 P1 统一重构为模块1/2/3（几何特征 -> 几何残差 -> 动作影响）。  
- `待追加`：每次里程碑或关键实验结论更新时追加一条。
