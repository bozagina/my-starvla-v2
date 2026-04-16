# CP-RES 分析请求：CP-AH-1 在 CorrectiveFlowHead 中的适用性

> 创建时间: 2026-04-15  
> 发起方: CP-BUILD  
> 目标方: CP-RES  
> 关联: cp_res_policy_rfc.md §6, corrective_flow_head.py L186-190

## 背景

CP-AH-1 定义（cp_res_policy_rfc.md §6）：
> trigger=0 样本的 region loss = 0（P0 生效验证），门限 100%

P0 机制在原始 corrective_loss pipeline（QwenPI.\_compute\_optional\_hook\_outputs 中
的 a\_module loss 路径）中已实现：region\_loss 仅在 trigger\_label > 0.5 的样本上计算。
500-step smoke train 中 `debug/corrective_loss_region_trigger_filtered_count = 8`
验证了原始 pipeline 的 P0 正常工作。

## 问题

CP-BUILD 新实现的 CorrectiveFlowHead（corrective\_flow\_head.py）有自己的
region-weighted loss，其计算逻辑如下：

```python
# corrective_flow_head.py L186-190
region_weight = torch.sigmoid(region_logits).unsqueeze(-1)  # [B, C, 1]
per_step_loss = (pred_velocity - velocity_gt) ** 2
base_loss = per_step_loss.mean()
region_loss = (region_weight * per_step_loss).mean()
loss = base_loss + self.region_loss_weight * region_loss
```

此处 region\_loss 对**全部样本**（包括 trigger=0）计算，没有 P0 门控。
输出的 `loss/corrective_flow` 是 base\_loss + 0.5 \* region\_loss 的聚合值。

## 需要 RES 分析的具体问题

### Q1. CP-AH-1 的适用范围

CP-AH-1 的 "region loss = 0 when trigger=0" 约束是否应当扩展到
CorrectiveFlowHead 内部的 region-weighted loss？还是 CP-AH-1 仅验证
原始 corrective\_loss pipeline 中的 `loss/region`？

### Q2. 语义合理性分析

在 CorrectiveFlowHead 中，trigger=0 样本的 region\_logits 来自
A-module 推理（detached）。对 trigger=0 样本：

- region\_logits 的语义是什么？（无需修正的样本，region 分布应该是什么？）
- 如果 region\_logits → 全负（sigmoid ≈ 0），则 region\_loss 自然趋零，
  P0 通过梯度隐式实现而非显式 mask——这是否可接受？
- 如果 region\_logits 非全负（sigmoid > 0），则 trigger=0 样本对
  corrective velocity 产生非零 region 加权监督——这是否有害？

### Q3. 建议修复方案（如需要）

如果 RES 判定需要在 CorrectiveFlowHead 中也实施 P0 门控，推荐的实现方式是什么？

- **方案 A**：在 forward() 中加 `trigger_mask = (trigger_prob > 0.5).float()`，
  用 trigger\_mask 零化 trigger=0 样本的 region\_loss
- **方案 B**：保持现状，依赖 A-module 的 region\_logits 在 trigger=0 时
  自然趋于负值（隐式 P0）
- **方案 C**：其他

### Q4. metrics 暴露

当前 `loss/corrective_flow` 是聚合值，无法从中分离出 base\_loss 和 region\_loss。
是否需要新增 debug metric `debug/corrective_flow_region_loss` 和
`debug/corrective_flow_base_loss` 以便独立验证？

## 可用数据

- 500-step metrics.jsonl（99 条记录）已拉取到本地
- `loss/corrective_flow` 范围 [0.085, 0.287]，均值 0.154，稳定下降
- 原始 `loss/region` 37 次为 0（P0 过滤生效），62 次非零
- `debug/corrective_loss_region_trigger_filtered_count` = 8（每 batch）

## 期望输出

- 对问题 Q1-Q3 的明确结论
- 如果结论是 "需要修复"，给出具体代码变更建议和新的 CP-AH-1 验证标准
- 如果结论是 "现状可接受"，给出理由并更新 CP-AH-1 的判定为 PASS（含说明）
