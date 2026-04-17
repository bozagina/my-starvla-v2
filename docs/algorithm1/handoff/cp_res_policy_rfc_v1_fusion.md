# Corrective Policy RFC v1.1 — RTC Paradigm + Fusion Mode

```
status:     APPROVED_BY_OPERATOR
author:     CP-RES thread
thread_id:  CP-RES
round_id:   CP-ROUND-FUSION-RESUME
gate:       phase3_0b_cp_fusion_resume
date:       2026-04-17
revision:   v1.1 (RTC paradigm revision)
supersedes: cp_res_policy_rfc.md (v0), cp_res_policy_rfc_v1_fusion.md (v1)
handoff_to: CP-BUILD, CP-REVIEW
```

---

## 0. 版本演变

| 版本 | 日期 | 核心变化 |
|------|------|---------|
| v0 | 2026-04-15 | 初版，lite mode，修正元数据预测（4 MLP heads）|
| v1 | 2026-04-17 早 | fusion mode 适配，BD-3 lite→fusion，保留 trigger/region 作为 CF 模型输入 |
| **v1.1** | **2026-04-17** | **RTC 范式修订：trigger/region 从 CF 模型输入中移除，仅保留为训练时 loss 权重；推理完全不依赖 A-module** |

### v1 → v1.1 修订动机

v1 中 trigger/region 同时作为 CF 模型的输入特征和 loss 权重，导致推理时必须运行
fusion head（需 VDPM, p95≈586ms）。这在部署上不可行，且造成训练/推理信号来源
不一致问题。v1.1 参考 RTC（Training-Time Action Conditioning）范式，将
trigger/region 的角色拆分为"仅训练时 loss 权重"，推理时 CF 仅依赖 a_prev。

---

## 1. 目标函数定义

### 1.1 问题描述

base policy（action chunk）在执行过程中产生状态偏差 $d_k$。
corrective policy 目标：**给定上一步预测 $a_{prev}$，预测一步修正 velocity field，
将动作从 $a_{prev}$ 修正到接近 $a_{gt}$**。

与 v0 的关键区别：CF 不再"给定 A-module 评估输出"做修正，而是从 $a_{prev}$ 本身
推断修正需求。A-module 信号仅在训练阶段用于引导学习。

### 1.2 形式化目标

设 $a_{prev} \in \mathbb{R}^{C \times D}$ 为上一步预测的 action chunk，
$a_{gt} \in \mathbb{R}^{C \times D}$ 为当前步 ground truth：

$$
v^* = a_{gt} - a_{prev} \quad \text{(target velocity)}
$$

CF 模型学习：

$$
\hat{v} = f_\theta(a_{prev},\ [o_t]) \quad \text{(predicted velocity)}
$$

其中 $o_t$（VLM features）为可选输入（由 BD-7 决定是否保留 cross-attention）。

修正后动作：

$$
a_{corrected} = a_{prev} + \hat{v}
$$

### 1.3 训练损失结构

```
corrective_flow_loss = base_loss + λ_region × region_loss

其中：
  base_loss   = mean( (v̂ - v*)² )                                   # 全样本
  region_loss = mean( sigmoid(region_logits) × (v̂ - v*)² )          # 全样本，region 加权
  region_logits 来自 fusion head（detached），仅用于 loss 加权
  λ_region = 0.5（默认）
```

**关键设计**：region_logits **不是** CF 模型的输入特征，仅影响 loss 梯度分布。
模型通过 region-weighted loss 学会在需要修正的时间步产生更大 velocity，
在不需要修正的时间步产生近零 velocity。

### 1.4 推理机制（RTC 范式）

```python
# 每步推理（无门控，无 A-module 依赖）
velocity = CF_model(a_prev)
a_corrected = a_prev + velocity
```

- **无 trigger 门控**：不再检查 trigger_prob > threshold
- **无 region gate**：不再用 `clamp(sigmoid(region) - 0.3)` 缩放 velocity
- **CF 每步都运行**：模型已通过训练学会在无需修正时输出近零 velocity
- **不依赖 A-module 推理**：不需要 fusion head / VDPM / trigger / region

### 1.5 A-module 信号在训练/推理中的角色

| 阶段 | A-module (fusion head) 角色 | CF 模型输入 |
|------|---------------------------|------------|
| **训练** | 运行 fusion head → 产出 region_logits → 仅用于 loss 加权 | a_prev（+ 可选 VLM features） |
| **推理** | **不需要运行**给 CF | a_prev（+ 可选 VLM features） |

---

## 2. CorrectiveFlowHead 架构修订

### 2.1 移除 MetadataEncoder 输入

| 组件 | v1（当前代码） | v1.1 (RTC) | 变化 |
|------|--------------|------------|------|
| MetadataEncoder | 编码 `cat(trigger_prob, region_logits)` → self-attn token | **从模型前向路径中移除**（可保留模块代码但不调用） | trigger/region 不再是模型输入 |
| self-attn tokens | `[meta_token, action_1..C]`（1+C tokens） | `[action_1..C]`（C tokens） | 减少 1 个 token |
| forward() 签名 | `(vl_embs, a_prev, a_gt, trigger_prob, region_logits)` | `(a_prev, a_gt, region_logits, vl_embs=None)` | trigger_prob 从签名移除；region_logits 仅用于 loss |
| predict() 签名 | `(vl_embs, a_base, trigger_prob, region_logits)` | `(a_prev, vl_embs=None)` | 极简推理签名 |

### 2.2 VLM cross-attention 决策（BD-7）

| 方案 | 描述 | 推荐度 |
|------|------|--------|
| **A: 保留 cross-attention** | DiT 对 VLM features 做 cross-attn，提供场景理解 | 可选（作为对比实验） |
| **B: 去掉 cross-attention** | `cross_attention_dim=None`，纯 self-attention | 推荐（更简洁，减少参数） |

**CP-BUILD 决策**：推荐 B（self-attention only），但可同时实现 A 作为消融实验。
若时间有限，优先实现 B。

### 2.3 固定 timestep 处理

当前 `_t_fixed = 0.9`，`_t_discretized = 900`。在 RTC 范式下 t 永远固定不变，
timestep embedding 退化为常量偏置。

**建议**：保留 timestep encoding 结构（兼容未来多步扩展），但 CP-BUILD 可选择
将 `num_embeds_ada_norm=1000` 降低或简化为 fixed bias。非关键，不阻塞。

---

## 3. QwenPI 集成修改

### 3.1 forward()（训练路径）

当前代码 `QwenPI.forward()` L510-549 中 CF 训练路径需修改：

```python
# ===== 修改前（v1 / 当前代码）=====
# L530: 调用 a_module_interface.predict() 获取 trigger/region 作为 CF 输入
a_predictions_cf = self.a_module_interface.predict(...)
trigger_prob = torch.sigmoid(a_predictions_cf.trigger_logit).detach()
region_logits = a_predictions_cf.region_logits.detach()
# L540: CF forward 将 trigger/region 作为模型输入
cf_loss, _, cf_debug = self.corrective_flow_head(
    vl_embs=vl_embs_for_cf,
    a_prev=a_prev_noisy,
    a_gt=actions_target,
    trigger_prob=trigger_prob,
    region_logits=region_logits,
)

# ===== 修改后（v1.1 RTC）=====
# region_logits 来自 fusion head 实时输出（已在 _compute_optional_hook_outputs 中计算）
# 仅用于 loss 加权，不作为 CF 模型输入
cf_loss, _, cf_debug = self.corrective_flow_head(
    a_prev=a_prev_noisy,
    a_gt=actions_target,
    region_logits=cached_region_logits.detach(),  # 仅用于 loss
    vl_embs=vl_embs_for_cf if use_cross_attn else None,
)
```

**ENG-1 解决方案**（训练路径信号来源统一）：
- fusion mode 下：从 `_compute_optional_hook_outputs` 缓存的 fusion head 输出获取 region_logits
- 非 fusion mode 下：从 `a_module_interface.predict()` 获取 region_logits
- 两种情况下 region_logits 都 **detach()** 后仅传给 loss 计算

### 3.2 predict_action()（推理路径）

```python
# ===== 修改前（v1 / 当前代码）=====
# L632: 调用 ami.predict() 获取 trigger/region
# L639: if trigger_prob > threshold → 启用修正
# L646: corrective_flow_head.predict(vl_embs, a_base, trigger_prob, region_logits)

# ===== 修改后（v1.1 RTC）=====
if self.corrective_flow_enabled:
    a_prev_for_cf = (
        self._cf_prev_chunk.to(device=pred_actions.device)
        if self._cf_prev_chunk is not None
        else torch.zeros_like(pred_actions)
    )
    a_corrected = self.corrective_flow_head.predict(
        a_prev=a_prev_for_cf.float(),
        vl_embs=base_hidden.float() if use_cross_attn else None,
    )
    normalized_actions = a_corrected.detach().cpu().numpy()
    self._cf_prev_chunk = pred_actions.detach().clone()
```

**ENG-2 彻底消除**：推理路径不再调用 `ami.predict()`，不依赖 fusion head / VDPM。

### 3.3 RNG 对齐（ENG-3）

```python
# 在 forward() 中 CF noise augmentation 使用独立 Generator
cf_rng = torch.Generator(device=a_prev.device)
cf_rng.manual_seed(42)  # 固定 seed 或从全局 seed 派生
a_prev_noisy = a_prev + noise_scale * torch.randn(
    a_prev.shape, device=a_prev.device, dtype=a_prev.dtype, generator=cf_rng
)
```

### 3.4 Episode reset（ENG-4）

`_cf_prev_chunk` 必须在新 episode 开始时重置为 None。CP-BUILD 需确保
`reset_cf_state()` 在 eval client 的 episode 循环中被正确调用。

---

## 4. BD 决策总览

| BD | 决策 | v0 | v1 | **v1.1 (最终)** |
|----|------|----|----|-----------------|
| BD-1 | 混合机制 | TBD | region-gated additive | **pure additive: a_prev + velocity** |
| BD-2 | 独立 CF head | 否 | 否 | 否 |
| BD-3 | A-module mode | lite | fusion | **训练 fusion，推理 CF 不依赖 A-module** |
| BD-4 | 消费 dynamic_embedding | 否 | 不由 CF 直接消费 | 不变 |
| BD-5 | 消费 region_target_15 | 否 | 否 | 不变 |
| BD-6 | CF 训练路径信号来源 | — | 待决 | **fusion head 实时输出（detached），仅用于 loss** |
| BD-7 | CF 架构 VLM cross-attn | — | 建议去掉 | **推荐去掉（self-attn only）；可做 A/B 消融** |
| **BD-8** | **trigger/region 角色** | — | — | **仅训练 loss 权重，不作为 CF 模型输入** |

---

## 5. 验收假设

### CP-AH-1 ~ CP-AH-3: 保持

| # | 指标 | 门限 |
|---|------|------|
| CP-AH-1 | trigger=0 的 region loss = 0（P0 验证） | 100% |
| CP-AH-2 | 500-step 所有 loss 有限且非退化 | 全通过 |
| CP-AH-3 | `a_loss_consist` 下降趋势 | 趋势下降 |

### CP-AH-4-v1: 多 seed 评估

```
Fusion CF model 的 Success@LIBERO 多 seed 均值
  ≥ Fusion Baseline 多 seed 均值 - 2pp

评估方式：
  1. Fusion Baseline: mode=fusion, a_loss + corrective_loss, 无 corrective_flow
     训练 N 步，seed={S1, S2, S3}，各跑 20 trials × 10 tasks
  2. Fusion CF (RTC): 同上 + corrective_flow_loss (RTC 范式) + CF 推理启用
     训练 N 步，seed={S1, S2, S3}，各跑 20 trials × 10 tasks
  3. 通过条件：CF 均值 ≥ Baseline 均值 - 2pp

远程验证：BLOCKED_WAIT_REMOTE
```

### CP-AH-5: RNG 对齐验证

```
目的：排除 RNG 分歧作为混淆变量
方法：CF forward 中使用独立 torch.Generator
验证：CF 模型关闭 CF 推理时 eval = Baseline eval（差 < 1pp）
```

---

## 6. 冻结锚点

| 锚点 | 值 | 本轮变化 |
|------|---|---------|
| VLM | qwen25_authoritative | 不变 |
| A-output contract | v0.9.3_frozen (fusion mode) | 不变 |
| schema | p1_shared_builder_v1 | 不变 |
| `optional_loss_utils.py` | 全文件冻结 | 不变 |
| `a_module_interface.py` | 冻结（现含 fusion 支持）| 不变 |
| `a_fusion_heads.py` | A-BUILD 产出，冻结 | 不变 |

---

## 7. 开放风险

| # | 风险 | 严重度 | 状态 |
|---|------|--------|------|
| CP-R1-v1 | CF 训练路径 region_logits 来源需统一为 fusion head 实时输出 | HIGH | 有具体方案（§3.1） |
| ~~CP-R2-v1~~ | ~~CF 推理路径未适配 fusion mode~~ | ~~HIGH~~ | **消除**（RTC 范式推理不依赖 fusion head）|
| ~~CP-R3-v1~~ | ~~VDPM 586ms 推理延迟~~ | ~~MEDIUM~~ | **消除**（推理不需要 VDPM）|
| CP-R4 | RNG divergence | MEDIUM | 有具体方案（§3.3 独立 Generator）|
| CP-R5 | VLM cross-attention 是否保留 | MEDIUM | BD-7 决策（推荐去掉）|
| CP-R7 | 模型能否学会自主判断修正幅度（无显式 trigger/region 输入） | LOW | TTAC 论文已验证；需本项目远程实验确认 |
| CP-R2 | correction_mask 零填充 50% | LOW | 继承，不变 |

---

## 8. 交付总结

| 工件 | 路径 |
|------|------|
| Policy RFC v1.1 | `docs/algorithm1/handoff/cp_res_policy_rfc_v1_fusion.md` |
| Interface Assumptions v1.1 | `docs/algorithm1/handoff/cp_res_interface_assumptions_v1_fusion.md` |
| Blocked Items v1.1 | `docs/algorithm1/handoff/cp_res_blocked_items_v1_fusion.md` |

**交付对象**：CP-BUILD、CP-REVIEW
