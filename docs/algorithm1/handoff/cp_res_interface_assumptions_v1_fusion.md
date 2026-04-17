# CP-RES Interface Assumptions Doc v1.1 — RTC Paradigm + Fusion Mode

```
status:     DELIVERED
author:     CP-RES thread
thread_id:  CP-RES
round_id:   CP-ROUND-FUSION-RESUME
gate:       phase3_0b_cp_fusion_resume
date:       2026-04-17
revision:   v1.1 (RTC paradigm revision)
supersedes: v0, v1
upstream:   A-REVIEW Round 5 PASS — fusion mode ACCEPTED, A-output USABLE
```

---

## 0. 上游稳定性声明

A-REVIEW Round 5 确认 fusion mode ACCEPTED, A-output USABLE_FOR_DOWNSTREAM。

---

## 1. CP 消费的 A-output 字段

### 1.1 训练期消费

| 类别 | 字段 | 用途 | 消费方式 |
|------|------|------|---------|
| **loss 权重** | `region_logits`（来自 fusion head 实时输出） | CF 的 region-weighted loss 加权 | `detach()` 后传给 `corrective_flow_head.forward()` 的 loss 计算 |
| **pseudo_labels** | `risk_score`, `trigger_label`, `delta_action_norm`, `correction_mask`, `affected_region_prior` | a_loss（corrective_loss）训练 | 经 `build_optional_hook_targets()` 转换为 tensor |

**关键变化（v1.1）**：region_logits **不再作为 CF 模型输入**，仅影响 loss 梯度分布。

### 1.2 推理期消费 — 极简

| 字段 | v1 | **v1.1 (RTC)** |
|------|----|----|
| trigger_logit | 需要（显式门控） | **不需要** |
| region_logits | 需要（region gate） | **不需要** |
| risk_pred | 需要 | **不需要**（可选用于监控日志） |
| delta_pred | 需要 | **不需要** |

**v1.1 推理时 CF 不消费任何 A-module 输出。** A-module 可独立运行用于安全监控。

### 1.3 不消费字段

与 v1 一致：`dynamic_embedding[16]`（fusion head 内部消费）、`region_target_15`、
`progress_t`、`intermediate_states`。

---

## 2. 消费接口绑定

### 2.1 训练期

- `optional_loss_utils.build_optional_hook_targets()`：冻结，不变
- CF 训练路径：从 `_compute_optional_hook_outputs()` 缓存的 fusion head 输出获取 region_logits

### 2.2 推理期

- **CF 不调用 `a_module_interface.predict()`**
- CF 推理仅需 `a_prev`（`_cf_prev_chunk`）和可选 VLM features
- `a_module_interface.py` 和 `a_fusion_heads.py` 冻结不变

---

## 3. CP-BUILD 接口决策

| 决策项 | v1.1 最终决策 |
|--------|-------------|
| A-module inference mode (训练) | fusion |
| A-module 推理时 CF 依赖 | **无依赖** |
| CF 训练 region_logits 来源 | fusion head 实时输出（detached） |
| CF 推理 trigger/region 依赖 | **无** |
| CF 架构 VLM cross-attn (BD-7) | 推荐去掉（CP-BUILD 决策） |
| RNG 对齐 (CP-AH-5) | 独立 torch.Generator |
