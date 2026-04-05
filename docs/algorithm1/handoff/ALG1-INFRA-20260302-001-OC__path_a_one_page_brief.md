# Path A 一页技术汇报（代码+公式版）

- 目标：用于你对内/对外汇报时，快速讲清 Path A 的实现原理、三模块状态、当前问题与解决路径。
- 口径范围：已整合 `EXP002 (ALG1-MASK-20260301-002-OC)` 与 `EXP003 (ALG1-MASK-20260301-003-OC)` 的代码与远程证据。

## 0) 来龙去脉（一段话）

从 `main` 分支 baseline 开始，我们最初是把视觉特征、几何特征和语言特征直接拼接后送入主干语言/动作建模链路，这个方案训练稳定但缺少显式时序纠偏信号；随后引入 Path A（早期 v3/v4）用双时刻 token 先 pool 成 `delta_z` 再生成 feedback token 注入，结果是“能跑且稳定”但增益偏弱（典型表现是 mask/slot 选择性接近均匀、反馈对主损失占比小），于是演进到 `v4-1`，在动作端加入有界残差 `a_out=a_base+gate*clip(DeltaA(feedback))`，成功点是稳定性和可控性提升，但失败点是早期仍未出现强选择性收益；接着进入 `v4-1-1`，把残差改成 token-level `Δgeo` 并叠加 soft-mask（含 directed/self-cross 与 teacher 监督），期间先后做了多轮关键尝试：一是验证 tower 假设（official/llava 重建后 alpha 余弦仍约 0.995+，因此这条假设失败并被排除），二是 directed 路径激活验证（成功接入但早期选择性提升不明显，属于部分成功），三是 m_clip teacher（首次因 `confidence_floor=0.02` 被门控导致 KL=0 属失败，降到 `0.002` 后 KL 与 weighted loss 非零属成功），四是 EXP003 中定位并修复 soft-mask 的 pre-split L2 压平问题为 per-head L2（显著成功，`entropy/topk/alpha_max/logits_std` 持续改善到长窗口），当前最新状态是：模块1表征构建稳定、模块2已成为主要有效增益来源但 teacher 分离度仍需提升、模块3注入有效但后段 clip saturation 偏高，我们正在用最小 CFG A/B 降饱和并继续校准 teacher（temperature/smoothing）以巩固已获得的长期收益。

---

## 1) Path A 当前要解决的核心

我们不是在重写动作模型，而是在现有 action 主干上增加“时序反馈纠偏”分支：

- 主目标：`action_dit_loss` 不退化（理想提升）。
- 并行目标：让 `soft-mask` 从高熵近均匀走向可解释、可选择。

总思路：`z_t(表征) -> r_t(残差) -> 注入利用(Δa)`。

---

## 2) 三模块实现（结合代码+公式）

### 模块1：`z_t` 多模态任务表征构建（输入基础）

**原理公式（对应固定 K task token）：**

- `q_task = LearnableQueries(K)`
- `c_geo = Attn(q_task, G_t, G_t)`
- `c_vis = Attn(q_task, V_t, V_t)`
- `s_lang = LangSummary(L_t)`
- `z_t = LN( MLP([c_geo, c_vis, s_lang]) + 0.5*(c_geo + c_vis) )`

**代码映射：**

- task token 构建：`/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:667` `def _build_fixed_task_tokens`
- 融合主入口：`/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:724` `def fusion_module`
- 时序 fast path：`/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:778` `def extract_task_tokens`

**当前状态：**

- 模块1工作稳定（无非数值异常）；不是当前主瓶颈。
- 主要问题不在“能否产出 token”，而在“残差信号如何被选择并有效注入”。
- 固定 `K` query 尝试本质归属模块1（表征构建稳定化）：
  - 代码上通过 `task_token_num` + `task_queries` 强制固定 token 数（默认 `K=32`），并由 `_build_fixed_task_tokens` 统一生成 `[B,K,H]`。
  - 该尝试在工程上是成功的（时序链路 shape 更稳定、便于 Path-A 双时刻对齐）；
  - 但在效果上属于“部分成功”：仅靠固定 `K` 并未带来明显选择性突破（历史观测 `slot_entropy≈ln(32)`，mask 仍偏高熵），因此它不是主增益来源。

---

### 模块2：`r_t` 残差构造 + soft-mask 选择（当前主战场）

**v4-1-1 主路径公式（token-level）：**

- `r_tokens = G_{t+k} - G_t`  （token-level 几何残差）
- `alpha = SoftMask(L_t, V_t, G_t)`
- `fb_vec = sum_j alpha_j * r_tokens_j`

**soft-mask 公式（多头聚合）：**

- `logits_vis = <Q(L), K(V)> * scale`
- `logits_geo = <Q(L), K(G)> * scale`
- `A = softmax(logits, token_dim)`
- `alpha = Agg_head( Agg_query(A) )`（含 channel 融合）

**teacher 监督公式（m_clip）：**

- `m_clip = softmax(cos(t_emb, patch_emb)/tau)`
- `L_teacher = E[ w(conf>=floor) * KL(m_clip || alpha_pred) ]`

**Residual 实际计算流程（代码口径）**

- 步骤1（长度对齐）：`token_n=min(N_geo_t,N_geo_tk,N_vis)`，并调用 `_resample_tokens_for_alignment` 保持全范围覆盖。
- 步骤2（残差构造）：`delta_tokens = geo_after - geo_before`，若 `causal_feedback_detach_delta=true` 则对 `delta_tokens` 截断梯度。
- 步骤3（mask 生成）：优先 `alpha=_build_soft_mask(...)`；若失败/shape不符/force_uniform，则 `alpha=1/N`。
- 步骤4（有效样本门控）：若存在 `valid_tk_mask`，对 `delta_tokens` 先做逐 batch 门控。
- 步骤5（汇聚）：`residual_tokens = delta_tokens * alpha[...,None]`，`fb_vec = sum(residual_tokens, dim=1)`。

**Soft-mask 实际计算流程（代码口径）**

- 步骤1：输入检查（dtype/ndim/batch/hidden 一致），然后对 `vision/geometric` 统一到 `token_n`。
- 步骤2：多头拆分 `q,v,g -> [B,h,T,dh]`。
- 步骤3：`l2_only` 下执行 **per-head** 归一化（当前关键修复点），随后计算：
  - `logits_vis = einsum(qh,vh) * scale`
  - `logits_geo = einsum(qh,gh) * scale`
- 步骤4：token 维 softmax 得 `attn_vis/attn_geo`，再做 `query_agg` 与 `head_agg`。
- 步骤5：按 `channel_mode` 融合 `alpha_vis/alpha_geo`，归一化得最终 `alpha`；可选 directed_self_cross 二次 refinement。

**代码映射：**

- 残差构造与模式切换：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:1525` `def _build_causal_feedback_tokens`
- soft-mask 构造：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:969` `def _build_soft_mask`
- token 对齐重采样：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:928` `def _resample_tokens_for_alignment`
- teacher 分布：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:470` `def _compute_soft_mask_teacher_distribution`
- teacher loss：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:571` `def _compute_soft_mask_teacher_loss`

**已起效果（强证据）：**

1. **per-head L2 修复后选择性显著提升（EXP003）**
   - 修复点：`soft_mask_score_norm=l2_only` 改为“split head 后归一化”（`_build_soft_mask`）。
   - 对照（common steps `19..1139`，新 run `...153623...` vs 旧 run `...135730...`）：
     - `soft_mask_entropy: 6.2368 -> 6.1982`
     - `soft_mask_topk_mass_32: 0.0712 -> 0.1105`
     - `soft_mask_alpha_max_mean: 0.00247 -> 0.00515`
     - `soft_mask_logits_std_vis: 0.0406 -> 0.4165`
   - 延长到 step `6499` 仍保持提升。

2. **teacher 从“接线成功但0贡献”变为“稳定非零监督”（EXP002）**
   - 问题 run（`...133009...`）：`confidence_floor=0.02` 导致 `sample_weight=0`, `KL=0`。
   - 调整后 run（`...135730...`，`floor=0.002`）：
     - `soft_mask_teacher_kl≈0.089~0.091`
     - `soft_mask_teacher_loss_weighted≈0.0018`
     - `sample_weight_mean=1.0`

**仍存在的问题（模块2内部）：**

- teacher 分离度仍偏弱（`neg_top1_gap` 小、confidence 量级低）。
- token mismatch 分支在部分 run 未触发，alignment 的泛化收益还需更多样本验证。

---

### 模块3：反馈注入与利用（action 端）

**注入公式（v4-1）：**

- `raw = DeltaA(mean(feedback_tokens))`
- `delta = clip * tanh(raw / clip)`
- `a_out = a_base + gate * delta`

其中 `gate` 由 schedule + valid_tk 控制，保证纠偏有界。

**代码映射：**

- delta-action 主逻辑：`/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1417` `def _apply_feedback_delta_action`
- context cross-attn 注入：`/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1171` `def _apply_layerwise_cross_attention`
- 动作上下文构造：`/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1108` `def build_world_action_context`

**已起效果：**

- 模块3路径稳定工作，`feedback_mask_contrast`、`delta_action_alpha` 等指标按计划激活。
- 结合 EXP003，任务主损失未见明显回归，同时 mask 选择性提升。

**当前瓶颈：**

- 后段 `delta_action_clip_saturation_frac` 偏高（说明纠偏经常撞 clip 上限），可能限制后续增益。

---

## 3) 现阶段问题定位：问题主要在哪个模块？

**结论：主问题在模块2，其次受模块3上限约束。**

- 模块1（表征构建）：可用且稳定，不是当前主因。
- 模块2（残差+mask）：历史上是主要问题来源（监督弱 + 实现尺度压平 + teacher门控），现在已获得明显改进但仍有 teacher 分离度问题。
- 模块3（注入利用）：机制有效，但 clip 饱和提示后段可能受上限约束。

---

## 4) 根本原因（当前统一结论）

根本原因不是视觉 tower 来源，而是：

1. **目标可辨识性不足**：mask 长期受间接监督，容易停在“平坦但可用”局部最优；
2. **实现细节曾压平 logits**：pre-split L2 使 head 级区分能力被系统性削弱；
3. **注入上限效应**：模块3后段 clip 饱和可能限制进一步收益释放。

---

## 5) 我们正在怎么解决（当前执行中）

1. **已完成并保留**：per-head L2 修复（模块2，已验证有效）。
2. **已完成并保留**：m_clip teacher 生效化（通过 floor 校准让 KL 非零）。
3. **已上线待验证**：token 对齐 range-resample + coverage 诊断（模块2）。
4. **下一步重点**：做最小 CFG A/B 降低模块3 clip saturation，并继续校准 teacher 分离度（temperature/smoothing）。

---

## 6) 汇报时可直接说的 3 句话

1. 我们已证伪“tower 主因”，并把问题收敛到模块2监督与实现细节。  
2. EXP003 的 per-head L2 修复带来可持续选择性提升，且任务损失未明显回归。  
3. 当前进入稳态优化阶段：主要解决模块3饱和上限和模块2 teacher 分离度。

