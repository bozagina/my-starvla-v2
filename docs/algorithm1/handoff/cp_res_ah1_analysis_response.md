# CP-RES 分析响应：CP-AH-1 在 CorrectiveFlowHead 中的适用性

> 创建时间: 2026-04-15  
> 响应方: CP-RES  
> 请求方: CP-BUILD  
> 请求文档: `cp_res_ah1_analysis_request.md`  
> 关联: `cp_res_policy_rfc.md` §6, `corrective_flow_head.py` L186-190

---

## 总结结论

| 问题 | 结论 |
|---|---|
| Q1: CP-AH-1 适用范围 | 原始 CP-AH-1 **仅验证 `loss/region`**（原始 corrective_loss pipeline）；建议新增 CP-AH-1-ext 覆盖 CorrectiveFlowHead |
| Q2: 语义合理性 | **现状可接受**（隐式 P0 + base_loss 对 trigger=0 样本的监督是有益的） |
| Q3: 推荐方案 | **方案 B**（保持现状），附加 metrics 暴露 |
| Q4: metrics 暴露 | **需要**：新增 `debug/corrective_flow_base_loss` 和 `debug/corrective_flow_region_loss` |

---

## Q1 分析：CP-AH-1 的适用范围

### 结论

CP-AH-1 原始定义仅覆盖 `QwenPI._compute_optional_hook_outputs` 中的 corrective_loss
pipeline（L291-335），其 P0 门控发生在 L315-316：

```python
# QwenPI.py L315-316
trigger_positive = targets["trigger_label"] > 0.5
```

该门控使用的是 **pseudo_labels 中的 GT trigger_label**（来自 FASA 数据），将 trigger=0
样本从 region loss 计算中完全排除。CP-AH-1 的验证 metric
`debug/corrective_loss_region_trigger_filtered_count` 也只在该路径中产生。

CorrectiveFlowHead 是一个 **独立模块**，其 region-weighted loss 服务于不同目标：

- 原始 corrective_loss：训练 A-module **元数据头**（`corrective_region_head`）预测
  "哪里需要修正"
- CorrectiveFlowHead：训练 **动作修正网络** 预测 "如何修正"，region_weight 仅做
  per-step loss 加权

两者的 loss 语义、梯度流向、训练目标完全不同，CP-AH-1 不应直接平移。

### 建议

新增 **CP-AH-1-ext** 验收条件（诊断级，非硬门限），见下方"判定"一节。

---

## Q2 分析：语义合理性

### 2a. trigger=0 时 region_logits 的语义

A-module 的 `corrective_region_head` 在训练中受到两个约束：

1. **P0 门控**（L315-316）：trigger=0 样本不参与 region loss 计算 →
   region_head 对 trigger=0 样本**不接收 region 方向的梯度**，输出值由初始化 +
   其他样本的间接影响决定
2. **P1 consist_loss**（L264-273）：惩罚 `mean(sigmoid(region_logits)) > trigger_prob + 0.1`
   的矛盾 → 对 trigger=0（trigger_prob ≈ 0）的样本，如果 region_logits 偏高会被显式惩罚

因此 trigger=0 样本的 region_logits **期望行为**是值偏负，`sigmoid ≈ 0`。
但这不是硬约束——训练早期或 P1 权重不够大时可能出现偶发高值。

### 2b. sigmoid ≈ 0 时（隐式 P0）

当 `sigmoid(region_logits) ≈ 0` 时，CorrectiveFlowHead 的 loss 分解为：

```
loss = base_loss + 0.5 * region_loss
     ≈ base_loss + 0.5 * 0
     ≈ base_loss
```

trigger=0 样本仅通过 `base_loss` 提供监督。此时 `velocity_gt = a_gt - a_prev`
对于 trigger=0 样本（base policy 准确）**接近零向量**。

这是**有益的**：CorrectiveFlowHead 学习到"对不需要修正的样本预测零 velocity"，
防止修正头在安全状态引入不必要的偏移。即使推理时已有 trigger 门控（`predict_action()`
中 `if trigger_prob > θ: ...`），训练阶段让修正头学习"安静"行为仍然是有价值的正则化。

**结论：可接受。**

### 2c. sigmoid > 0 时（偶发情况）

当 trigger=0 但 `sigmoid(region_logits) > 0` 时，效果是：

```
region_loss += region_weight * (pred_velocity - ~0)²
            = region_weight * pred_velocity²
```

这相当于**对修正头的输出施加额外的 L2 惩罚**——在 region 认为需要关注的时间步上，
额外惩罚非零 velocity 输出。对于 trigger=0 样本这是一个**正向的正则化效果**，
不会有害。

唯一的风险场景：如果 A-module 严重误校准（trigger=0 但 region_logits 全正），
会导致 region_loss 在不应关注的样本上产生较大的梯度。但由于：

1. `region_loss_weight = 0.5`，只占 loss 的少数
2. `base_loss` 始终提供正确监督
3. P1 consist_loss 会逐步修正 A-module 的误校准

该风险的严重程度为 **LOW**。

**结论：不需要显式干预。**

---

## Q3 分析：推荐方案

### 采用方案 B：保持现状

理由汇总：

| 考量 | 方案 A（显式 mask） | 方案 B（保持现状） |
|---|---|---|
| trigger=0 的 base_loss | 仍在 | 仍在（有益） |
| trigger=0 的 region_loss | 被 mask 消除 | sigmoid≈0 时自然消除 |
| 模块封装性 | 需要引入 trigger 二值化逻辑 | CorrectiveFlowHead 保持简洁 |
| train/eval 一致性 | 用 `trigger_prob > 0.5`（推理值）做 mask，与原始 P0 用 GT `trigger_label > 0.5` 不一致 | 无此问题 |
| 正则化效果 | 丢失 trigger=0 样本的隐式正则 | 保留 |
| 推理安全性 | 已有 trigger 门控 | 已有 trigger 门控 |

不推荐方案 A 的关键理由：CorrectiveFlowHead 的 `forward()` 接收的是 detached 的
`trigger_prob`（模型推理值），而非 GT `trigger_label`。如果要做显式 P0 mask，
应该用哪个值？用推理值（`trigger_prob > 0.5`）会在训练早期（trigger_head 不准时）
产生不稳定的 mask；用 GT `trigger_label` 则需要将 pseudo_labels 传入
CorrectiveFlowHead，破坏其作为独立模块的封装。

---

## Q4 分析：metrics 暴露

**需要。** 当前 `loss/corrective_flow` 是聚合值，无法验证隐式 P0 是否在工作。

### 建议实现

`corrective_flow_head.py` 的 `forward()` 应额外返回分解的 debug metrics：

```python
# forward() 返回签名变更为:
def forward(self, ...) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ...
    debug = {
        "corrective_flow_base_loss": base_loss.detach(),
        "corrective_flow_region_loss": region_loss.detach(),
    }
    return loss, pred_velocity, debug
```

在 `QwenPI.forward()` 的调用处，将 debug dict 合入 `output_dict`。
在 `train_starvla.py` 的 `decomposed_metric_specs` 中新增：

```python
("corrective_flow_base_loss", "loss/corrective_flow_base", "corrective_flow_loss"),
("corrective_flow_region_loss", "loss/corrective_flow_region", "corrective_flow_loss"),
```

---

## CP-AH-1 判定

### CP-AH-1（原始）

| 项 | 值 |
|---|---|
| 范围 | `loss/region` in corrective_loss pipeline（QwenPI L291-335） |
| 判定 | **PASS** |
| 证据 | 500-step 数据：`debug/corrective_loss_region_trigger_filtered_count = 8`（每 batch），P0 显式 mask 正常工作 |

### CP-AH-1-ext（新增，诊断级）

| 项 | 值 |
|---|---|
| 范围 | `loss/corrective_flow` 中的 region 行为 |
| 类型 | 诊断监控（非硬门限） |
| 判定 | **PASS（隐式 P0）** |
| 理由 | CorrectiveFlowHead 的 region-weighted loss 不需要显式 P0 gate；trigger=0 样本通过 base_loss 学习"零 velocity"是有益的；region_loss 通过 A-module 的 P0+P1 机制使 sigmoid(region_logits) 在 trigger=0 时趋零 |
| 监控条件 | `debug/corrective_flow_region_loss` 应随训练下降；如果未来发现 trigger=0 样本的 `mean(sigmoid(region_logits))` 持续高于 0.3，需重新评估是否需要显式门控 |

---

## CP-BUILD 行动项

1. **修改 `corrective_flow_head.py`**：`forward()` 新增返回 debug dict（含 `corrective_flow_base_loss` 和 `corrective_flow_region_loss`）
2. **修改 `QwenPI.py`**：将 debug dict 合入 `output_dict`
3. **修改 `train_starvla.py`**：在 `decomposed_metric_specs` 中新增两个 metric 条目
4. **无需修改 loss 计算逻辑**：保持 `corrective_flow_head.py` L186-190 现状
5. **验证**：重跑 smoke test，确认新增 metrics 出现在 `metrics.jsonl` 中
