# Path A 端到端总说明文档（整合版）

- 文档目标：把 Path A 从设计原点到当前实验结论完整串联，形成一份可直接用于后续整理/汇报/交接的总文档。
- 整合来源：
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/path_a_llm_review_packet.md`
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260302-001-OC__report.md`
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md`
- 对应整理任务：`ALG1-INFRA-20260302-001-OC`

---

## 0. 一句话总览

Path A 当前的核心突破不是“换视觉塔”，而是“把 mask 从弱间接监督推进到可辨识监督，并修复会系统性压平注意力的实现细节（per-head L2）”；这条路线在远程运行中已经出现持续的选择性提升证据。

---

## 1. 为什么从 Path A 开始

### 1.1 Path A 的任务定位

Path A 的目的不是替代主动作模型，而是给动作预测提供“由时序变化构造的反馈条件”。

直观上：

- 主分支负责“当前该做什么”；
- Path A 负责“基于前后变化，怎么纠偏更合理”。

### 1.2 初始症状

项目早期观察到：

- Path A 接入后训练稳定；
- 但相对 baseline，`action_dit_loss` 并未持续表现出明显更快收敛。

因此评审重点转为三问：

1. 残差信号是否被正确构造？
2. 注入方式是否被主干真正使用？
3. 损失是否真的在约束 mask/feedback，而非只“存在但弱影响”？

---

## 2. Path A 版本演化脉络（必须统一口径）

### v3/v4（旧路径）

- 从 `[B,K,H]` token 先 pool 成 `[B,H]`，得到 `delta_z`。
- 再映射为 feedback token 注入动作头。
- 主要问题：pool 过早，可能丢失 token 级空间差异。

### v4-1

- 增加动作侧 `Δa` 残差头：`a_out = a_base + gate * clip(DeltaA(feedback))`。
- 优势：注入有上界、稳定性更好。
- 但仍可能存在“反馈信号强度不足或被忽略”。

### v4-1-1（当前主线）

- 残差主路径升级为 token-level `Δgeo`，再用 soft-mask 做语言条件筛选。
- 与 v4-1 组合：保留有界 delta-action 注入。
- 核心目标：
  - 模块2（残差构造）更保真；
  - 模块3（注入利用）更稳健可控。

---

## 3. 算法框架原理（端到端）

> 本节是“从输入到损失到推理”的总流程。

### 3.1 训练阶段总流程

1. `t` 时刻经过 VLM 提取 token：
   - `task_tokens`
   - `vision_tokens`
   - `geometric_tokens`
2. `t+k` 时刻再提取对照 token（训练中通常 no-grad）。
3. 构造残差（旧：`delta_z`；新：`delta_geo_tokens`）。
4. 构造语言条件 soft-mask `alpha`。
5. 用 `alpha` 筛选残差并汇聚成 feedback 表征。
6. 将 feedback 注入 action head（context 或 delta-action 分支）。
7. 联合主损失 + feedback/mask/teacher 辅助损失反传。

### 3.2 核心张量与形状（统一口径）

- `task_tokens`: `[B, K, H]`（常见 `K=32`, `H=4096`）
- `task_tokens_next`: `[B, K, H]`
- `vision_tokens`: 常见 `[B, 512, H]`（双视角拼接）
- `geometric_tokens(_next)`: 常见 `[B, 512, H]`
- `feedback_tokens`: `[B, Kf, H]`（常见 `Kf=4`）
- `actions_target`: `[B, T, A]`（常见 `A=7`）

### 3.2.1 固定 K query 尝试归因（容易被遗漏）

这次尝试本质属于模块1（`z_t` 表征构建），不是模块2/3：

- 实现方式：用 `task_token_num` 与可学习 `task_queries` 固定 task token 数（默认 `K=32`），由 `_build_fixed_task_tokens` 统一生成 `[B,K,H]`，并在时序路径复用。
- 代码锚点：
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:133`（`task_token_num`）
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:136`（`task_queries`）
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:667`（`_build_fixed_task_tokens`）
- 为什么要做：减少可变长度 token 带来的时序对齐噪声，稳定 `t/t+k` 残差构造输入。
- 结果判断：
  - 工程层面成功（shape 稳定、链路更可控）；
  - 算法增益层面部分成功（仅此改动未打破低选择性：历史上 `slot_entropy≈ln(32)`，mask 仍高熵），因此不是当前主突破点。

### 3.3 soft-mask 构造机制

输入：`q`（语言 query）、`v`（视觉 token）、`g`（几何 token）。

流程：

1. token 对齐到统一长度 `N`（当前已支持范围保真重采样）。
2. 多头拆分并计算相似度 logits。
3. token 维 softmax -> query/head 聚合 -> 通道融合。
4. 得到 `alpha: [B,N]`，并归一化为概率分布。

用途：`alpha` 作为 token 级权重，决定哪些时序残差更值得进入动作纠偏。

### 3.4 m_clip teacher 监督机制

动机：只靠动作主损失对 `alpha` 是“间接监督”，容易得到“可用但偏平”的局部最优。

做法：

- 用 SigLIP 文本-视觉检索分布构造 teacher soft label `m_clip`；
- 优化 `KL(m_clip || alpha_pred)`；
- 通过 `confidence_floor` 门控样本权重；
- 加入 shuffle 文本负控指标验证 teacher 不是“伪信号”。

### 3.5 动作侧注入机制（v4-1 保留）

- 主体：`a_base`
- 纠偏：`DeltaA(feedback)`
- 约束：`clip(tanh)` + `gate(schedule)`
- 最终：`a_out = a_base + gate * clipped_delta`

意义：在不破坏稳定性的前提下引入反馈增量。

### 3.6 推理闭环

推理期维护上一帧 token 与动作 chunk；当前帧自动构造 feedback 后再预测下一 chunk，实现在线闭环纠偏。

---

## 4. 损失与优化信号：到底谁在推动什么

可抽象为：

- `L_total = L_action + L_feedback_aux + L_mask_contrast + L_teacher + ...`

其中：

- `L_action`：动作主任务（最终性能主导）。
- `L_feedback_aux`：约束 feedback 残差重构/方向一致性。
- `L_mask_contrast`：增强 masked vs unmasked 的可区分性。
- `L_teacher`：给 `alpha` 的直接外部监督。

关键认识：

- 若 `L_teacher`/contrast 权重、门控或实现细节不当，mask 会停在高熵“可用但不尖锐”的区域。

---

## 5. 问题树：我们是如何定位根因的

### H1：是不是视觉塔 fallback 导致 mask 偏平？

- 触发点：日志出现 SigLIP fallback。
- 验证：`current vs llava_rebuild vs official_ckpt` 三分支对比。
- 结果：alpha 分布几乎一致（cos≈0.995~0.996）。
- 结论：**非主因**。

### H2：是不是结构（directed_self_cross）就能直接改善？

- 动机：用更强 token 关系建模提升选择性。
- 结果：路径激活成功，但早期 matched window 选择性没有明显优于 similarity。
- 结论：结构可用，但**不是单点突破**。

### H3：是不是 teacher 监督根本没生效？

- 首次 teacher run：`active=1` 但 `sample_weight=0`, `KL=0`。
- 原因：`confidence_floor=0.02` 高于实际 confidence 量级（~0.005）。
- 修正：降到 `0.002`。
- 结果：KL/weighted loss 非零，teacher 生效。
- 结论：问题在门控阈值，不在接线逻辑。

### H4：是否存在实现层面的“系统性压平”bug？

- 动机：监督路径已有效，但选择性增益仍受限。
- 发现：`l2_only` 在 head split 前归一化，缩小每头有效幅度。
- 修复：改成 per-head 归一化。
- 结果：远程 run 选择性指标显著提升且持续。
- 结论：这是关键增益点。

### H5：token 对齐是否有前缀偏置风险？

- 动机：`token_n=min(Nv,Ng)` + prefix 截断可能导致覆盖偏差。
- 改动：替换为范围保真重采样 + coverage 诊断。
- 现状：代码与指标已接入，需更多 mismatch 样本验证收益。

---

## 6. EXP002（ALG1-MASK-20260301-002-OC）完整复盘

### 6.1 做了什么

1. 建立可追溯基础：修 fetch 匹配、确认 canonical branch、run identity 对齐。
2. 比较 similarity 与 directed_self_cross。
3. 接入 m_clip teacher 与可靠性验证脚本。
4. 排查 teacher 门控并修正 confidence floor。

### 6.2 为什么这样做

- 先保证“拿到的是对的 run”，否则后续分析无效。
- 再验证结构改动是否立竿见影。
- 若结构收益有限，转向直接监督（teacher）。

### 6.3 结果与结论

- directed 路径：可运行但早期未显著提升 mask 选择性。
- teacher 路径：
  - 初次被门控静默关闭；
  - floor 降低后稳定生效（KL 与 weighted loss 非零）。
- EXP002 产出：
  - 方向从“单纯结构尝试”转向“监督+实现细节并行优化”。

---

## 7. EXP003（ALG1-MASK-20260301-003-OC）完整复盘

### 7.1 做了什么

1. 系统低选择性代码审计。
2. 修复 soft-mask per-head L2 归一化。
3. 长窗口远程对照验证（含 step 6499）。
4. 增加 token 对齐与 coverage 诊断。

### 7.2 为什么这样做

- EXP002 后确认“监督方向正确但增益不够强”，需要排查实现层面缩放问题。
- 同时补齐潜在 token 对齐偏置风险，为后续泛化打底。

### 7.3 关键结果

相对旧 run（common steps `19..1139`）显著改善：

- `soft_mask_entropy`: `6.2368 -> 6.1982`
- `soft_mask_topk_mass_32`: `0.0712 -> 0.1105`
- `soft_mask_alpha_max_mean`: `0.00247 -> 0.00515`
- `soft_mask_attn_top1_mean_vis`: `0.00232 -> 0.00657`
- `soft_mask_logits_std_vis`: `0.0406 -> 0.4165`

且在 step `6499` 仍保持提升趋势。

### 7.4 结论

- per-head L2 修复是当前已被远程证据验证的关键改动。
- EXP003 将主线从“是否可行”推进到“如何稳态优化”。

---

## 8. 当前被证实的事实 vs 仍待验证项

### 8.1 已被证实

1. tower mismatch 不是当前主矛盾。
2. teacher 监督可生效（阈值合理时）。
3. per-head L2 修复可显著提升选择性并持续。
4. 训练稳定性总体可控（远程指标未见 nonfinite 问题）。

### 8.2 仍待验证

1. 后段 `delta_action_clip_saturation_frac` 偏高是否成为瓶颈。
2. teacher 负控分离（如 `neg_top1_gap`）是否能随训练或校准持续增强。
3. token mismatch 分支在更多数据条件下的真实收益。

---

## 9. 统一解释：为什么这条路线成立

这条路线的核心逻辑是“先排错因，再增强可辨识监督，再修实现细节”。

- 若不先排 tower，会在错误方向耗费训练预算。
- 若不补 teacher，mask 很容易停在高熵但可用的平坦解。
- 若不修 per-head L2，注意力尺度会被系统性压平，监督效果也被削弱。

因此当前结果并非偶然调参，而是有因果链支持的工程-算法协同结果。

---

## 10. 后续执行建议（可直接转任务）

### P0

1. 保持当前关键修复不回滚（per-head L2 + teacher 生效配置）。
2. 进行最小 CFG A/B，目标降低 `delta_action_clip_saturation_frac`。
3. 固定 2k/5k/10k 评估窗口，统一比较 `mask + action + teacher` 三类指标。

### P1

1. 小范围 teacher 校准（temperature/smoothing）提升分离度。
2. 在触发 mismatch 的数据上验证 token 对齐重采样收益。

### P2

1. 产出“管理版 1 页摘要 + 技术版详报”双套文档模板。
2. 自动化 run 对照报告，减少手工汇总成本。

---

## 11. 关键代码与文档索引

- Path A 评审总包：
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/path_a_llm_review_packet.md`
- 当前详报（含 EXP002/EXP003）：
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260302-001-OC__report.md`
- 本文（总整合版）：
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260302-001-OC__path_a_end_to_end_guide.md`
- 实时进度：
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md`

---

## 12. 对外汇报可直接使用的结论

- Path A 路线没有推翻重来，而是在原框架内完成了可追溯、可解释、可验证的增量演进。
- 当前最强证据来自 EXP003：mask 选择性提升显著且具有长窗口持续性。
- 下一阶段重点是“保持增益并解除后段饱和与 teacher 分离度瓶颈”，而不是回到 tower 来源争论。

