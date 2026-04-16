# CP-RES 分析请求：CP-AH-4 LIBERO 评测失败根因分析

> 创建时间: 2026-04-15  
> 发起方: CP-BUILD  
> 目标方: CP-RES  
> 优先级: **P0** — 阻塞 Corrective Flow 方案可行性判定  
> 关联文档: cp_ah4_libero_results.md, cp_res_ah4_eval_protocol.md, cp_res_policy_rfc.md

---

## 1. 背景与实验设置

### 1.1 CP-AH-4 验收标准

> Success@CF ≥ Success@Baseline - 2pp

### 1.2 实验配置


| 参数                    | Baseline                       | CF                                               |
| --------------------- | ------------------------------ | ------------------------------------------------ |
| 起始检查点                 | 2d_vlm_no_geo_30k              | 2d_vlm_no_geo_30k (同一个)                          |
| A-module              | 启用 (lite mode)                 | 启用 (lite mode)                                   |
| CorrectiveFlowHead    | **禁用**                         | **启用** (hidden_dim=1024, region_loss_weight=0.5) |
| corrective_flow_scale | N/A                            | 0.5                                              |
| 额外训练步数                | 3000                           | 3000                                             |
| 训练配置                  | starvla_train_pi_from30k.yaml  | starvla_train_pi_from30k.yaml                    |
| LIBERO 评测             | libero_goal, 20 trials, seed=7 | libero_goal, 20 trials, seed=7                   |


### 1.3 结果

**CP-AH-4 判定: FAIL** — CF 62.0% < Baseline 69.0% - 2pp = 67.0%

---

## 2. 需要 RES 分析的核心问题

### Q1. CF 模型退化根因诊断

CF 整体成功率比 Baseline 低 7pp，但**分布极不均匀**：


| Task | Baseline | CF      | Delta     | 特征      |
| ---- | -------- | ------- | --------- | ------- |
| T0   | 60%      | **85%** | **+25pp** | CF 大幅改善 |
| T6   | 50%      | **60%** | **+10pp** | CF 改善   |
| T4   | 100%     | 100%    | 0pp       | 饱和      |
| T9   | 5%       | 5%      | 0pp       | 均失败     |
| T3   | 35%      | 30%     | -5pp      | 轻微退化    |
| T1   | 100%     | 90%     | -10pp     | 从饱和退化   |
| T7   | 100%     | 90%     | -10pp     | 从饱和退化   |
| T8   | 100%     | 90%     | -10pp     | 从饱和退化   |
| T5   | 45%      | 15%     | **-30pp** | 严重退化    |
| T2   | 95%      | 55%     | **-40pp** | 严重退化    |


**关键模式**: CF 在困难任务上改善，但在"中高难度"任务上严重退化。请 RES 分析以下假设：

- **H1 (训练不足)**: 3000 步是否不足以让 CorrectiveFlowHead 充分收敛？CF 的 `loss/corrective_flow` 从 0.174 降到 0.008，看似收敛，但推理时 correction 质量可能仍不够。
- **H2 (损失权重干扰)**: `corrective_flow_scale=0.5` 加上 `loss/corrective`、`loss/a_module` 等多头损失，是否对 base action 模型产生了有害梯度干扰？
- **H3 (trigger 质量差)**: 推理时 trigger head 从随机初始化训练 3000 步，其 P(trigger) 预测质量如何？错误的 trigger 预测会导致不该修正时强行修正。
- **H4 (correction 方向错误)**: 对于 T2/T5 这类严重退化的任务，correction velocity 是否在把动作推向错误方向？
- **H5 (简单任务过修正)**: T1/T7/T8 从 100% 降到 90%，是否因为 CF 在已经正确的动作上进行了不必要的"修正"？

### Q2. Baseline 异常高性能来源

我们的 Baseline (69%) 比同一 30k 检查点在旧代码下的评测 (~48%, 24/50) 高出约 20pp。请分析这一差异的来源：

- **因素 A**: 额外 3000 步训练的贡献
- **因素 B**: A-module 辅助损失 (risk/trigger/embed) 的多任务学习/正则化效应
- **因素 C**: 评测协议差异（5 trials vs 20 trials, 不同 seed, 不同代码路径）
- **因素 D**: 新代码中 predict_action() 的修改（state truncation, dtype handling 等）

如果 A-module 辅助损失本身就带来了显著的性能提升（因素 B），这对 Corrective Flow 方案的意义是什么？是否说明：

- 辅助损失的正则化效果已经足够强，CF 的额外收益有限？
- 还是说 CF 有独立的价值，只是需要更多训练步数来体现？

### Q3. 推理路径分析

请审查 `QwenPI.predict_action()` 中 corrective flow 的推理路径（L561-600）：

```python
if self.corrective_flow_enabled:
    with torch.autocast("cuda", enabled=False):
        # ... cast to float, get A-module predictions ...
        a_predictions_cf = ami.predict(pooled_hidden=pooled_cf, examples=examples)
        trigger_prob = torch.sigmoid(a_predictions_cf["trigger_logit"])
        region_logits = a_predictions_cf["region_logit"]
        # ... apply CorrectiveFlowHead correction ...
```

具体问题：

1. `trigger_prob` 是否有阈值门控？还是对所有样本都应用 correction？
2. correction 的幅度是否受到合理约束？
3. `region_logits` 对修正的引导是否正确？

### Q4. 40k 检查点不兼容问题

40k ablation baseline（旧代码训练）在新代码推理下产生 NaN/Inf (0% success)。30 个缺失 key 均为 A-module/corrective 头。请分析：

- 这是纯粹的 `strict=False` 随机头问题？
- 还是新代码的 `predict_action()` 路径本身（即使 `corrective_flow_enabled=False`）也对旧检查点不兼容？
- 是否需要实现"旧检查点兼容模式"以获得有效的 40k 参考数据？

### Q5. 下一步实验建议

基于以上分析，请给出：

1. **最小干预方案**: 仅调超参数/训练步数，不改架构，需要做哪些实验？
2. **推理消融方案**: 保持 CF 训练但推理时禁用 correction（验证 CF 训练是否损害了 base model）
3. **架构修复方案**: 如果分析发现架构问题，推荐的具体代码修改
4. **判定标准更新**: 基于当前结果，CP-AH-4 是否需要调整验收标准或实验设计？

---

## 3. 完整训练损失数据

### 3.1 Baseline (30k→33k, 600 entries, 每5步记录一次)


| Metric        | s30001 | s30301 | s30751 | s31501 | s32251 | s32996 |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| loss/total    | 11.500 | 0.898  | 0.237  | 0.149  | 0.307  | 0.086  |
| loss/action   | 0.134  | 0.171  | 0.065  | 0.082  | 0.305  | 0.055  |
| loss/a_module | 11.313 | 0.277  | 0.068  | 0.006  | 0.001  | 0.001  |
| loss/risk     | 0.400  | 0.006  | 0.002  | 0.002  | 0.000  | 0.000  |
| loss/trigger  | 0.742  | 0.219  | 0.027  | 0.001  | 0.000  | 0.000  |
| loss/embed    | 10.188 | 0.052  | 0.039  | 0.003  | 0.001  | 0.001  |


### 3.2 CF (30k→33k, 600 entries, 每5步记录一次)


| Metric                      | s30001 | s30301 | s30751 | s31501 | s32251 | s32996 |
| --------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| loss/total                  | 11.375 | 0.918  | 0.213  | 0.144  | 0.262  | 0.083  |
| loss/action                 | 0.107  | 0.144  | 0.055  | 0.067  | 0.250  | 0.047  |
| loss/a_module               | 11.000 | 0.336  | 0.041  | 0.008  | 0.001  | 0.001  |
| loss/corrective             | 0.190  | 0.424  | 0.099  | 0.063  | 0.000  | 0.031  |
| loss/corrective_flow        | 0.174  | 0.025  | 0.036  | 0.012  | 0.020  | 0.008  |
| loss/corrective_flow_base   | 0.139  | 0.023  | 0.034  | 0.010  | 0.020  | 0.007  |
| loss/corrective_flow_region | 0.070  | 0.005  | 0.004  | 0.002  | 0.001  | 0.000  |
| loss/delta                  | 0.190  | 0.017  | 0.099  | 0.001  | 0.000  | 0.000  |
| loss/region                 | 0.000  | 0.406  | 0.000  | 0.062  | 0.000  | 0.031  |
| loss/risk                   | 0.381  | 0.011  | 0.004  | 0.005  | 0.000  | 0.000  |
| loss/trigger                | 0.742  | 0.252  | 0.015  | 0.000  | 0.000  | 0.000  |
| loss/embed                  | 9.875  | 0.073  | 0.022  | 0.002  | 0.001  | 0.001  |


### 3.3 训练损失对比观察

- **loss/action**: CF (0.047) 略低于 Baseline (0.055) — CF 训练未明显损害 base action
- **loss/a_module**: 两者均收敛到 ~0.001 — A-module 学习无差异
- **loss/corrective_flow**: 0.174→0.008 — 下降 95%，但最终值仍非零
- **loss/region 波动**: CF 的 `loss/region` 在某些采样点为 0（batch 无 trigger>0 样本），其他点 0.03-0.4 — 高方差

---

## 4. LIBERO 评测数据

### 4.1 Per-Task 完整结果

详见 `cp_ah4_libero_results.md`。

### 4.2 历史参考


| 评测                 | 代码版本 | 检查点                  | 协议                | 成功率             |
| ------------------ | ---- | -------------------- | ----------------- | --------------- |
| 旧代码 30k baseline   | 旧    | 2d_vlm_no_geo_30k    | 5 trials, 旧 seed  | ~48% (24/50)    |
| 新代码 Baseline (33k) | 新    | 30k+3k (A-module)    | 20 trials, seed=7 | 69.0% (138/200) |
| 新代码 CF (33k)       | 新    | 30k+3k (A-module+CF) | 20 trials, seed=7 | 62.0% (124/200) |
| 新代码加载 40k          | 新    | 旧代码 40k ckpt         | 20 trials, seed=7 | 0.0% (NaN)      |


---

## 5. 可用 Artifact 路径（远程服务器）


| Artifact               | Path                                                      |
| ---------------------- | --------------------------------------------------------- |
| Baseline 检查点           | `results/Checkpoints/cp_ah4_baseline_3k_20260415_081523/` |
| CF 检查点                 | `results/Checkpoints/cp_ah4_cf_3k_20260415_104250/`       |
| Baseline metrics.jsonl | 同上 (600 条)                                                |
| CF metrics.jsonl       | 同上 (600 条)                                                |
| Baseline eval log      | `results/LiberoEval/cp_ah4_baseline_eval/eval.log`        |
| CF eval log            | `results/LiberoEval/cp_ah4_cf_eval/eval.log`              |
| 40k eval log           | `results/LiberoEval/cp_ah4_40k_eval/eval.log`             |
| 训练配置                   | `starVLA/config/training/starvla_train_pi_from30k.yaml`   |
| CF 推理代码                | `starVLA/model/framework/QwenPI.py L561-600`              |
| CorrectiveFlowHead     | `starVLA/model/framework/corrective_flow_head.py`         |


---

## 6. 期望输出

1. **Q1 根因诊断**: 对 H1-H5 五个假设的逐一判定（likely/unlikely），并给出最可能的根因排序
2. **Q2 Baseline 高性能**: 分析 20pp 提升来源，量化各因素贡献估计
3. **Q3 推理路径**: 审查 predict_action() 中 CF 路径的正确性，指出任何可能的 bug 或设计缺陷
4. **Q4 兼容性**: 40k NaN 问题的根因和修复建议
5. **Q5 下一步**: 具体实验方案（含超参数建议、步数建议、消融设计）
6. **方案可行性判定**: 基于当前证据，Corrective Flow 方案是否值得继续投入？建议 CONTINUE / PIVOT / PAUSE

