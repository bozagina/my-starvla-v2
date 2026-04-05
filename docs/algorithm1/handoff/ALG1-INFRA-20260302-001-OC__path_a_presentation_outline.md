# Path A 汇报PPT提纲（实现细节版，12页）

- 对应整理任务：ALG1-INFRA-20260302-001-OC
- 建议受众：技术评审 / 研发同步 / 跨团队汇报
- 核心材料：
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260302-001-OC__path_a_one_page_brief.md`
  - `/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/ALG1-INFRA-20260302-001-OC__path_a_end_to_end_guide.md`

---

## Slide 1. 从 baseline 到现在（一页讲清来龙去脉）

- `main` baseline：视觉/几何/语言直接拼接送主干，稳定但无显式时序纠偏。
- Path A v3/v4：pooled `delta_z` 反馈，能跑但增益弱。
- v4-1：加 `Δa` 有界注入，稳定性提升。
- v4-1-1：token-level `Δgeo + soft-mask + teacher`，开始获得持续选择性提升。

## Slide 2. baseline（main 分支）机制与局限

- 机制：`X=[V,G,L] -> backbone -> action`。
- 优点：结构简单、训练稳定。
- 局限：缺少“前后时刻差异”的显式通路，纠偏能力弱、可解释性弱。

## Slide 3. Path A 总框架：三模块

- 模块1：`z_t` 多模态任务表征构建。
- 模块2：`r_t` 残差构造与 soft-mask 选择。
- 模块3：反馈注入与动作端有界利用（delta-action）。
- 总式：`z_t -> r_t -> a_out = a_base + gate * clip(DeltaA(feedback))`。

## Slide 4. 模块1实现（代码+公式）

- 公式：
  - `q_task=LearnableQueries`
  - `c_geo=Attn(q_task,G_t,G_t)`
  - `c_vis=Attn(q_task,V_t,V_t)`
  - `z_t=LN(MLP([c_geo,c_vis,s_lang])+0.5*(c_geo+c_vis))`
- 代码：
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:667`
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:724`
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:778`
- 结论：模块1稳定，不是当前主瓶颈。

## Slide 5. 模块2实现A：Residual 具体怎么算

- 主路径（v4-1-1）：
  - `r_tokens = G_{t+k} - G_t`
  - `fb_vec = sum(alpha_j * r_tokens_j)`
- 关键实现步骤：
  1) `token_n=min(N_geo_t,N_geo_tk,N_vis)`
  2) `_resample_tokens_for_alignment` 对齐
  3) `delta_tokens=geo_after-geo_before`
  4) 乘 `alpha` 并按 token 求和
- 代码：
  - `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:1525`
  - `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:928`

## Slide 6. 模块2实现B：Soft-mask 具体怎么算

- 公式：
  - `logits_vis=<Q(L),K(V)>*scale`
  - `logits_geo=<Q(L),K(G)>*scale`
  - `A=softmax(logits, token)`
  - `alpha=Agg_head(Agg_query(A))`
- 实现要点：
  - 多头拆分后执行 per-head L2（关键修复）
  - 可选 `directed_self_cross` refinement
  - fallback: 输入缺失/shape 不符 -> uniform `1/N`
- 代码：
  - `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:969`

## Slide 7. 模块2实现C：Teacher 监督（m_clip）

- 公式：`L_teacher = KL(m_clip || alpha_pred)` with confidence gate。
- 首次失败：`confidence_floor=0.02` 导致 `sample_weight=0`, `KL=0`。
- 修正成功：`confidence_floor=0.002` 后 `KL/weighted_loss` 持续非零。
- 代码：
  - `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:470`
  - `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:571`

## Slide 8. 模块3实现：反馈注入与有界纠偏

- 公式：
  - `raw=DeltaA(mean(feedback_tokens))`
  - `delta=clip*tanh(raw/clip)`
  - `a_out=a_base+gate*delta`
- 代码：
  - `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1417`
  - `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1171`
- 现状：注入路径有效，但后段 clip saturation 偏高。

## Slide 9. EXP002：尝试、成功与失败

- 成功：
  - 修复 run 追溯链（fetch/run_identity/branch）。
  - teacher 从“接线存在”推进到“非零监督”。
- 失败/部分失败：
  - directed_self_cross 早期窗口未显示明显选择性优势。
- 结论：方向从“结构优先”转为“监督+实现细节并行”。

## Slide 10. EXP003：关键突破与证据

- 突破点：soft-mask pre-split L2 -> per-head L2。
- 对照结果（common steps `19..1139`）：
  - entropy 下降、topk/alpha_max/logits_std 显著上升。
- 长窗口到 step `6499` 仍保持提升趋势。

## Slide 11. 目前问题定位与根本原因

- 问题定位：
  - 主问题在模块2（mask/teacher 可辨识与分离度）
  - 次问题在模块3（后段 clip 饱和）
- 根本原因：
  1) 历史上 mask 直接监督不足
  2) 实现细节曾压平 logits
  3) 注入上限限制后段收益释放

## Slide 12. 当前方案与下一步

- 已保留：per-head L2 + teacher 生效配置。
- 已上线待验证：token 对齐重采样 + coverage 诊断。
- 下一步：最小 CFG A/B 降低 saturation + teacher 温度/平滑校准。
- 保留标准：选择性持续提升且 `action_dit_loss` 不退化。

---

## 附：30秒开场口播（可选）

我们从 main baseline 的三特征直接拼接起步，系统稳定但没有显式时序纠偏；Path A v3/v4 证明了反馈可接入但增益偏弱，v4-1 解决了注入稳定性，v4-1-1 进一步把残差升级到 token-level 并加入 soft-mask 与 teacher。经过 EXP002/EXP003，我们排除了 tower 主因，确认当前主要矛盾在模块2监督与实现细节，且 per-head L2 修复已带来持续可复现的选择性提升，下一步聚焦模块3饱和与 teacher 分离度优化。

