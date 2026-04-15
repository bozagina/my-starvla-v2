# CP-RES Interface Assumptions Doc

```
status:     DELIVERED
author:     CP-RES thread
thread_id:  CP-RES
round_id:   CP-ROUND-BOOTSTRAP-WAIT-UPSTREAM
gate:       phase3_0b_cp_research
date:       2026-04-15
upstream:   a_outputs_current_reviewed (A-REVIEW PASS_CONFIRMED v0.9.3)
```

---

## 0. 上游稳定性声明

A-output contract v0.9.3 已由 A-REVIEW 发布 **PASS_CONFIRMED + downstream_usability=USABLE**
（reviewed_at: 2026-04-15T00:30:00+08:00）。

CP 线程族已完全解封，以下所有字段均基于已验证合同。

---

## 1. CP 消费的 A-output 字段

### 1.1 训练期消费（来自 `pseudo_labels` dict in FASA records）

| 字段 | 来源键名 | 类型 | 形状 | 稳定性 | 说明 |
|---|---|---|---|---|---|
| `risk_score` | `pseudo_labels.risk_score` | float | scalar | **STABLE** | clip(max(f_H, d_H), 0, 1)；A-REVIEW 验证通过 |
| `trigger_label` | `pseudo_labels.trigger_label` | float (0 or 1) | scalar | **STABLE** | 𝟙[f_H=1 or d_H > α_d]；自适应 α_d 标定；A-REVIEW 验证通过 |
| `delta_action_norm` | `pseudo_labels.delta_action_norm` | float | scalar | **STABLE** | outcome-mixed scalar：max(max_j‖Δa_j‖, β×d_H)，β=0.5；A-REVIEW F-5 已修正为 scalar |
| `correction_mask` | `pseudo_labels.correction_mask` | list[float] | [horizon_steps=4] | **STABLE** | 𝟙[d_k > α_d or f_H=1]；v0.9.3 逐步状态偏差；零填充至 chunk_len=8 by `_coerce_vector` |
| `affected_region_prior` | `pseudo_labels.affected_region_prior` | list[float] | [horizon_steps=4] | **STABLE** | d_k 归一化分布 + conditional γ 叠加；v0.9.3；零填充至 chunk_len=8 |

### 1.2 推理期消费（来自 `AModulePredictions`，当 mode=standalone 时）

| 字段 | 接口键名 | 类型 | 来源 |
|---|---|---|---|
| `risk_pred` | `a_outputs.risk_pred` 或 `risk_score` | float | StandaloneAModuleInterface._extract_source |
| `trigger_logit` | `a_outputs.trigger_logit` 或 `trigger_label` | float (logit) | 同上，fallback: prob → logit |
| `delta_pred` | `a_outputs.delta_pred` 或 `delta_action_norm` | float | 同上 |
| `region_logits` | `a_outputs.region_logits` 或 `affected_region_prior` | list[float] | `_coerce_vector` → chunk_len |

### 1.3 当前不消费的 A-output 字段

| 字段 | 原因 | 未来状态 |
|---|---|---|
| `dynamic_embedding[16]` | 在 `a_outputs` dict 中可用，但 FASA builder 不生成；仅通过 `a_outputs.dynamic_embedding` 传递 | 若 CP-BUILD 需要，需在 Round 2 明确 |
| `region_target_15` | 15-bin 重采样字段，由分析使用，当前不参与训练 | CP-BUILD 可选择消费；需在 RFC 中声明 |
| `intermediate_states` | 用于 pseudo-label 构建，不直接传入训练 tensor | 未来可作为 corrective policy 额外输入 |
| `progress_t` | 轨迹进度信号；FASA record 字段，不在当前训练 pipeline 中消费 | 可选未来输入 |
| `_outcome_v3` | 仅诊断字段，不参与训练 | 永久诊断用途 |

---

## 2. 消费接口绑定

### 2.1 训练期：`optional_loss_utils.build_optional_hook_targets()`

```
输入：examples (list[dict])，每个 dict 含 pseudo_labels 和可选 a_outputs
输出：targets dict，含以下 tensor：
  - risk_score    [B]        + risk_mask    [B, bool]
  - trigger_label [B]        + trigger_mask [B, bool]
  - delta_action_norm [B]    + delta_mask   [B, bool]
  - correction_mask   [B, C] + correction_mask_mask [B, bool]
  - region_prior      [B, C] + region_prior_mask    [B, bool]
  - dynamic_embedding [B, 16] + embedding_mask      [B, bool]

其中 C = chunk_len（训练时通常 = 8）
```

> **冻结约束**：`optional_loss_utils.py` 全文件冻结，CP-BUILD 不得修改。

### 2.2 推理期：`a_module_interface.build_a_module_interface()`

```
mode=lite     → LiteAModuleInterface（pooled_hidden → 4 lightweight heads）
mode=standalone → StandaloneAModuleInterface（从 example["a_outputs"] 读取外部 A-module 预测）
```

> **冻结约束**：`a_module_interface.py` 全文件冻结，CP-BUILD 不得修改。

---

## 3. 零填充边界声明（OBS-2 继承）

`correction_mask` 和 `affected_region_prior` 在 v0.9.3 中长度为 `horizon_steps=4`，  
由 `_coerce_vector` 零填充至 `chunk_len=8`（positions 5-8 receive zero supervision）。

**语义含义**：观测窗口之外的动作步无修正监督，视为"无纠偏"。  
**影响**：有效监督率 = 4/8 = 50%；在 AH-4 v0.9.3 中已验证不影响训练稳定性。  
**CP-BUILD 注意**：delta loss masking 建议仅对 mask=True 的样本计算，已由 `delta_mask` 提供。

---

## 4. 字段稳定性分类总结

### STABLE（CP-BUILD 可直接使用，无需等待进一步确认）

- `risk_score`
- `trigger_label`
- `delta_action_norm`（scalar）
- `correction_mask`（len=4，零填充至 chunk_len）
- `affected_region_prior`（len=4，零填充至 chunk_len）
- 接口：`optional_loss_utils.build_optional_hook_targets()`
- 接口：`a_module_interface.build_a_module_interface()`

### PROVISIONAL（存在但 CP 侧未声明消费）

- `dynamic_embedding[16]`：可用，CP-BUILD 需显式声明是否消费
- `region_target_15`：可用，CP-BUILD 需显式声明是否消费

### OUT_OF_SCOPE for v0

- 实际动作偏移向量 Δa（FASA 中无直接存储，需 base policy rollout 才能推导）
- `intermediate_states` 作为推理输入
- `progress_t` 作为推理输入

---

## 5. CP-BUILD 需确认的接口决策

CP-BUILD 在实现 v0 时需明确以下决策，并在其交付物中声明：

| 决策项 | 选项 | 默认建议 |
|---|---|---|
| 是否消费 `dynamic_embedding[16]` | 是 / 否 | 否（v0 简化） |
| 是否消费 `region_target_15`（15-bin） | 是 / 否 | 否（当前接口用 chunk_len-sized region） |
| 推理时 action 混合机制 | 加法混合 / 条件替换 / 门控混合 | 待 CP-BUILD 设计 |
| A-module inference mode | lite / standalone | lite（v0 默认） |
| delta loss masking 策略 | trigger=1 only / all samples | trigger=1 only（per RFC） |
