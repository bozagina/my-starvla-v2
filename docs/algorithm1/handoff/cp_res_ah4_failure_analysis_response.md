# CP-RES 分析响应：CP-AH-4 LIBERO 评测失败根因分析

```
响应方:     CP-RES 线程
请求方:     CP-BUILD
日期:       2026-04-15
优先级:     P0
关联文档:   cp_res_ah4_failure_analysis_request.md, cp_ah4_libero_results.md
审查范围:   QwenPI.py L561-597, corrective_flow_head.py, base_framework.py
```

---

## 执行摘要

CP-AH-4 失败的根因不是训练不足，而是**推理路径存在一个基础设计缺陷**（H4）和一个**触发控制问题**（H5）。核心问题是：CorrectiveFlowHead 的训练目标与推理语义存在系统性不匹配，导致在已收敛的任务上引入噪声。该问题可以通过代码修复解决，不需要重新设计整体方案。

**建议：PAUSE（暂停）**，先以零额外训练成本做推理消融，确认根因后执行代码修复，再重跑 CP-AH-4。

---

## Q1. CF 模型退化根因诊断

### H1（训练不足）— **LIKELY，但非首要根因**

**支持证据**：

- `loss/corrective_flow_base` 在第 33k 步仍为 0.007（非零），说明 velocity 预测未完全收敛
- FASA 2000 samples 仅经历约 3.75 epochs

**反驳证据**：

- `loss/trigger` 和 `loss/a_module` 均已收敛至 ~0（<0.001），A-module heads 已充分学习
- 训练步数不足无法解释高度非均匀的 per-task 模式（T0 +25pp vs T2 -40pp）
- 如果只是训练不足，期望是均匀的小幅退化，而非某些任务严重退化、某些大幅改善

**结论**：H1 在一定程度上成立（更多训练会减少噪声），但非 T2/T5 严重退化的根因。

---

### H2（损失权重干扰）— **UNLIKELY**

**支持证据**：None

**反驳证据**：

- CF 模型 `loss/action`（0.047）实际上**低于** Baseline（0.055）
- 两者 `loss/a_module` 完全相同（~0.001）
- CF 总损失（0.083）≈ Baseline（0.086），损失量级相当

**结论**：H2 可以排除。`corrective_flow_scale=0.5` 的额外损失未对 base action 模型产生有害干扰。

---

### H3（trigger 质量差）— **LIKELY，辅助因素**

**支持证据**：

- 仅 3000 步 + 2000 FASA samples 训练 trigger head
- `loss/region` 在 CF 训练中呈现高方差（0.000 ↔ 0.406），说明每个 batch 中 trigger>0.5 的样本比例波动大
- 在 LIBERO rollout 环境中，状态偏差分布与 FASA 离线标注数据分布不同，导致 trigger 预测存在域偏移

**反驳证据**：

- 仅靠 trigger 质量差无法解释 T0/T6 的显著改善（trigger 应该同样在这些任务上产生噪声）
- 如果 trigger 主要是问题，期望是 CF 模型均匀地引入 10-15pp 的噪声，而非出现 +25pp/−40pp 的极端分化

**结论**：H3 是放大器，在 H4/H5 基础上加剧了问题，但单独不能解释 T2/T5 的严重退化。

---

### H4（correction 方向错误）— **VERY LIKELY，T2/T5 退化首要根因**

这是代码审查发现的**最关键设计缺陷**，详见 Q3。

**核心问题**：CorrectiveFlowHead 训练时接收 `a_prev`（GT 前序 chunk），预测 `velocity_gt = a_gt - a_prev`。但推理时接收 `a_base`（当前基础策略预测），这两者有根本性的语义差异。

**数学分析**：

训练期间 CF head 学到的映射：

```
f(vl_embs, a_prev, trigger, region) → velocity ≈ a_gt - a_prev
```

推理期间期望 CF head 完成的任务：

```
correction = a_gt - a_base_policy
```

这两个任务等价的条件是：`a_prev ≈ a_base_policy`，即"基础策略的预测恰好类似于前序 chunk"。

**为什么 T0 改善而 T2 退化**：

- T0（开抽屉）：任务涉及重复性到达动作，基础策略的预测具有一定时间连续性，`a_base ≈ a_prev`，CF head 的速度预测方向接近正确修正
- T2（Baseline 95%）：高成功率任务意味着基础策略在执行精确、动态的动作序列，相邻 chunk 差异显著，`a_base ≫ a_prev` 的偏差导致 CF head 预测出错误方向的速度

**定量支持**：

- T2: Baseline 95% → CF 55%，损失 8/20 成功 episodes
- T5: Baseline 45% → CF 15%，损失 6/20 成功 episodes
- 这两个任务的共同特征：Baseline 已经有相当高的成功率，说明基础策略在这些任务上执行有效的主动轨迹规划（a_base 与 a_prev 差异大）

**结论**：H4 是 T2/T5 严重退化的首要根因。

---

### H5（简单任务过修正）— **VERY LIKELY，T1/T7/T8 退化首要根因**

**支持证据**：
T1、T7、T8 从 100% 下降到 90%，**完全一致的 -10pp**，这不是随机噪声，是系统性行为。

分析 `loss/corrective_flow_region = 0.0003`：这意味着 region_logits 已经收敛到非常小（甚至负）的值，`sigmoid(small_negative) ≈ 0.4-0.5`。

在推理时：

```python
region_gate = sigmoid(region_logits)  # ≈ 0.4-0.5 for near-zero logits
a_corrected = a_base + region_gate * pred_velocity
           ≈ a_base + 0.45 * pred_velocity
```

而 `loss/corrective_flow_base = 0.007`，即 velocity 的 RMS 幅度约 `sqrt(0.007) ≈ 0.084`（在归一化 action 空间）。7 个动作维度的 correction RMS 约 `0.084 * 0.45 * sqrt(7) ≈ 0.10`，对于已经正确的轨迹，这是不可忽视的噪声。

**T1/T7/T8 均为先前 100% 成功的任务**：trigger 在这些任务上偶尔触发（约每 10 个 episode 触发一次），将 0.10 量级的噪声引入已经完美的轨迹，偶尔导致失败。三个任务一致的 -10pp 是概率性的系统性噪声，而非任务特定问题。

**结论**：H5 是 T1/T7/T8 退化的首要根因。

---

### 根因优先级排序


| 优先级   | 假设                     | 影响任务                   | 判定                     |
| ----- | ---------------------- | ---------------------- | ---------------------- |
| **1** | H4：a_prev/a_base 语义不匹配 | T2 (-40pp), T5 (-30pp) | **VERY LIKELY，代码设计缺陷** |
| **2** | H5：trigger 激活时的系统性过修正  | T1/T7/T8 (各 -10pp)     | **VERY LIKELY，推理控制缺陷** |
| **3** | H3：trigger 质量差（域偏移）    | 全部任务，放大效果              | **LIKELY，辅助因素**        |
| **4** | H1：训练步数不足              | 全部任务，轻微                | **LIKELY，但非首要**        |
| **5** | H2：损失权重干扰              | —                      | **UNLIKELY，可排除**       |


---

## Q2. Baseline 异常高性能来源（69% vs 历史 ~48%）

20pp 的提升来自多个叠加因素，按贡献估计排序如下：

### 因素 B：A-module 多任务学习正则化效应 — **估计 +8~12pp（最大贡献）**

训练损失表格清晰展示了 `loss/a_module` 从 11.313 下降到 0.001，是整个训练过程中下降最剧烈的分量。A-module heads（risk/trigger/embed）的联合训练对 VLM backbone 特征空间施加了强多任务约束，充当了隐式正则化：

- `loss/embed`：10.188 → 0.001，这是动作 embedding 质量的强监督信号
- `loss/trigger`：0.742 → 0.000，迫使 VLM 提取精确的 state deviation 判别特征
- 这些额外损失提升了 VLM 特征对机器人动作状态的表达能力，从而改善了 base action policy

### 因素 C：评测协议差异 — **估计 +5~8pp（统计噪声 + 协议变化）**

- 旧代码：5 trials × 10 tasks = 50 episodes，95% CI 约 ±14pp（$\sigma \approx \sqrt{0.5 \times 0.5 / 50} \approx 7$）
- 新代码：20 trials × 10 tasks = 200 episodes，95% CI 约 ±7pp
- 48% 有可能是 50 episode 下的低侧采样（真实性能可能为 55-60%）
- seed 差异和初始状态分布差异也贡献了部分差距

### 因素 A：额外 3000 步训练 — **估计 +4~6pp**

从损失曲线看，Baseline 的 `loss/action` 在训练过程中从 0.134 下降到 0.055，约降低 60%。额外的梯度更新提升了 action model 的精度。这是真实的性能提升，但幅度低于 B 和 C。

### 因素 D：新代码 predict_action 改进 — **估计 +1~3pp**

- state truncation（8→7）的正确实现
- bf16/float32 dtype 处理改进
- 推理路径的整体稳定性

### 对 CF 方案意义的解读

A-module 辅助损失（因素 B）**已经带来了独立的性能提升**，这具有重要含义：

1. **A-module 本身就有价值**，即使没有 CF head 的推理修正，多任务正则化也显著改善了基础策略
2. **CF 的额外收益必须在 A-module 已经优化的基线上体现**，竞争门槛更高
3. **这不意味着 CF 无价值**：T0 +25pp 的改善说明在正确工作时 CF 有真实效果，只是当前推理路径存在 H4/H5 问题阻碍了整体收益

---

## Q3. 推理路径审查（predict_action() L561-597）

### 发现 1：a_prev/a_base 语义不匹配 — **[CRITICAL BUG]**

**训练时**（`forward()` L462-484）：

```python
# a_prev = 从 GT 动作序列取前一个 chunk（+ 噪声）
if actions.shape[1] >= 2 * self.chunk_len:
    a_prev = actions[:, :self.chunk_len, :]
else:
    a_prev = torch.zeros_like(actions_target)
a_prev_noisy = a_prev + noise_scale * torch.randn_like(a_prev)

cf_loss, _, cf_debug = self.corrective_flow_head(
    a_prev=a_prev_noisy,   # ← 前序 GT chunk
    a_gt=actions_target,   # ← 当前 GT chunk
    ...
)
# CF head 学到：velocity_gt = a_gt - a_prev
```

**推理时**（`predict_action()` L583-589）：

```python
a_corrected = self.corrective_flow_head.predict(
    a_base=pred_actions.float(),  # ← 当前基础策略预测，≠ a_prev
    ...
)
# CF head 看到 a_base，但尝试预测"如何修正它"
```

**缺陷本质**：CF head 的内部映射是 `(vl_embs, a_prev, trigger, region) → velocity ≈ a_gt - a_prev`。在推理时它看到 `a_base` 作为输入，但 `a_base` 与 `a_prev` 不同，所以输出的 velocity 没有正确的参考框架。

对任务的影响：

- 当 `a_base ≈ a_prev`（如重复性动作任务）：velocity 方向正确，CF 有益 → T0 +25pp
- 当 `a_base ≠ a_prev`（如已收敛的精确任务）：velocity 方向错误，CF 有害 → T2 -40pp, T5 -30pp

**推荐修复**：在推理路径中维护 `prev_action_chunk` 状态（即上一步推理返回的 chunk），作为 `a_prev` 传入 CF head：

```python
# 在模型状态中维护:
self._prev_action_chunk = None  # 在 reset() 时清空

# 在 predict_action() 中:
a_prev_for_cf = self._prev_action_chunk if self._prev_action_chunk is not None \
    else torch.zeros_like(pred_actions)
a_corrected = self.corrective_flow_head.predict(
    a_base=pred_actions.float(),  # 改为: a_input=a_prev_for_cf.float()
    ...
)
self._prev_action_chunk = pred_actions.detach()
```

这使推理语义与训练语义完全对齐。

---

### 发现 2：region_gate 无下限，导致全 chunk 低幅度噪声 — **[DESIGN ISSUE]**

```python
# corrective_flow_head.py predict():
region_gate = torch.sigmoid(region_logits).unsqueeze(-1)  # ≈ 0.4-0.5 when logits≈0
a_corrected = a_base + region_gate * pred_velocity
```

当 `loss/corrective_flow_region → 0`（当前 0.0003），region_logits 趋向小值，`sigmoid(near_zero) ≈ 0.5`。这意味着即使"低 region 相关性"的 chunk 步骤，也会收到约 50% 幅度的速度修正。

**推荐修复**：在 predict() 中添加阈值化或幅度限制：

```python
# 方案 A：region 阈值化
region_gate = (torch.sigmoid(region_logits) > 0.5).float().unsqueeze(-1)

# 方案 B：更保守的 scaling（避免在 region 低时产生修正）
region_gate = torch.clamp(torch.sigmoid(region_logits) - 0.3, min=0.0).unsqueeze(-1)
```

---

### 发现 3：trigger 门控是全 chunk 二元开关 — **[DESIGN OBSERVATION]**

```python
# predict_action() L581:
if trigger_prob.max().item() > self.corrective_flow_trigger_threshold:
    # ... 对 ENTIRE chunk 应用修正
```

trigger 是每 chunk 一次的全局决策：要么全修正，要么完全不修正。配合上述 region_gate 问题，当 trigger 误触发时，整个 chunk 都会被以约 50% 幅度的 velocity 扰动。

**短期缓解**：提高 trigger_threshold（如 0.8-0.9）以减少误触发频率。

---

### 发现 4：base_hidden shape 传递正确 ✓

训练和推理路径均使用 `vl_embs_list[-1]`（最后一层 VLM hidden states，`[B, S, hidden_dim]`），传递给 CF head 的 `vl_embs` 来源一致。无 bug。

### 发现 5：trigger_prob 维度处理正确 ✓

训练路径：`torch.sigmoid(logit).detach().unsqueeze(-1)` → `[B, 1]`  
推理路径：`torch.sigmoid(logit).unsqueeze(-1)` → `[B, 1]`  
两者一致，`_encode_and_attend()` 的 `torch.cat([trigger_prob, region_logits], dim=-1)` 可以正确拼接。

---

## Q4. 40k 检查点 NaN 根因分析

### 根因诊断

**结论：不是纯粹的 `strict=False` 随机头问题；根因是 predict_action() 内部不寻常的代码路径与旧 ckpt 权重的组合。**

分层分析：

**层 1：`strict=False` 加载的随机头是否直接导致 NaN？**

从代码路径看，`predict_action()` 中 LiteAModuleInterface（A-module 随机头）只在 `if self.corrective_flow_enabled:` 块内被调用（L561）。如果旧 ckpt config 不含 `framework.corrective_flow` 字段，则 `corrective_flow_enabled=False`，该块不执行，随机 A-module 头不参与推理。

**层 2：action_model 路径是否有潜在不兼容？**

40k ckpt 用旧代码训练，旧代码的 action_model 权重应能被新代码加载（键名相同）。但 `vl_embs_list` 的提取方式可能不同：

```python
# predict_action() L536-538
all_hidden = qwenvl_outputs.hidden_states
expected_layers = len(self.action_model.model.transformer_blocks)
vl_embs_list = list(all_hidden[-expected_layers:])
```

如果旧 ckpt 的 action_model 有不同的 `transformer_blocks` 数量（或该属性不存在），`expected_layers` 计算可能出错，导致取到错误的 hidden states → action_model 接受了形状或数值不对的输入 → NaN。

**层 3：norm_stats 的 unnorm_key 问题**

`model2libero_interface.py` 的 `get_action_stats()` 调用 `_check_unnorm_key(norm_stats, unnorm_key=None)`，会 assert `len(norm_stats) == 1`。如果 40k ckpt 在多数据集上训练，`norm_stats` 有多个 key，这里会 assert 失败。但这应该是 exception 而非 NaN。

**最可能的根因**：旧 ckpt 的 `config.yaml` 中 `action_model.transformer_blocks` 属性与新代码期望的名称不匹配，导致 `expected_layers` 错误，VLM hidden states 传递错误的层到 action_model，产生 NaN/Inf。

### 修复建议

1. **短期（不改代码）**：不要使用 40k 旧ckpt + 新代码评测。用新代码训练的 Baseline（30k+3k）作为对比基线，已经满足 CP-AH-4 实验需求。
2. **如需 40k 参考数据**：用新代码重新训练一个 "30k ckpt + 10k 步，无 CF"的 ablation，这是更干净的对比。
3. **代码兼容性修复**：在 `from_pretrained()` 后增加兼容性检查，对 `expected_layers` 等关键参数做存在性断言。

---

## Q5. 下一步实验建议

### 方案 A：推理消融（最优先，零额外训练成本）

**实验**：用已训练的 CF 33k ckpt，在 eval 时禁用 CF 修正路径（令 `trigger_threshold=1.1` 或直接跳过 corrective_flow 块），对比：

```
A1: CF ckpt + CF 推理禁用 vs Baseline ckpt（均在 libero_goal, 20 trials）
```

**判读逻辑**：


| 结果                         | 结论                           | 下一步                                 |
| -------------------------- | ---------------------------- | ----------------------------------- |
| CF-推理禁用 ≈ Baseline（67-72%） | CF 训练不损害 base model；问题纯在推理路径 | → 修复推理路径（方案 C）                      |
| CF-推理禁用 < Baseline（< 65%）  | CF 训练轻微损害 base model         | → 同时修复推理 + 调小 corrective_flow_scale |
| CF-推理禁用 > Baseline（> 72%）  | A-module 联合训练有额外好处（意外发现）     | → 研究原因                              |


**时间**：1-2 小时 eval（无需训练）

---

### 方案 B：trigger 阈值消融（零额外训练，快速）

```
B1: CF ckpt, trigger_threshold=0.7
B2: CF ckpt, trigger_threshold=0.8
B3: CF ckpt, trigger_threshold=0.9
```

验证 H5 的量化影响：高阈值 → T1/T7/T8 是否恢复到 100%？

**判读逻辑**：如果 threshold=0.9 时 T1/T7/T8 均回到 100%，而 T0/T6 改善幅度缩小，则确认 H5；同时为最终阈值选择提供数据。

**时间**：3×1.5 小时 eval ≈ 4-5 小时

---

### 方案 C：推理路径修复（代码修改 + 重跑 eval）

修复发现 1（a_prev/a_base mismatch）和发现 2（region_gate 无下限）：

**C1：维护 prev_action_chunk 状态**

在 `predict_action()` 中维护上一次推理的 action chunk，作为 CF head 的 `a_input` 而非当前基础策略预测：

```python
# ModelServer 或 predict_action() 层维护:
self._cf_prev_chunk: Optional[torch.Tensor] = None  # 在 episode reset 时清空

# predict_action() 中:
a_prev_for_cf = (self._cf_prev_chunk if self._cf_prev_chunk is not None 
                 else torch.zeros_like(pred_actions)).float()
a_corrected = self.corrective_flow_head.predict(
    vl_embs=base_hidden.float(),
    a_base=a_prev_for_cf,     # ← 使用前一 chunk（与训练语义对齐）
    trigger_prob=trigger_prob.unsqueeze(-1).float(),
    region_logits=region_logits.float(),
)
self._cf_prev_chunk = pred_actions.detach()  # 保存本次 chunk 供下次使用
```

**注意**：需要在 eval 脚本的 episode reset 处调用 `model.reset_cf_state()` 以清空 `_cf_prev_chunk`。

**C2：region_gate 阈值化**

```python
# corrective_flow_head.py predict():
region_gate = torch.clamp(torch.sigmoid(region_logits) - 0.3, min=0.0).unsqueeze(-1)
```

这将 `sigmoid(logit) < 0.3` 的步骤完全关闭（不修正），避免全 chunk 低幅度噪声。

**时间**：代码修改约 30 分钟 + 1.5 小时 eval（无需重跑训练，直接对 CF 33k ckpt 执行修复后的推理）

---

### 方案 D：增加训练步数（可选，在 C 修复后）

如果 C 修复后仍有 gap，再考虑增加步数（10k）。调整建议：

```yaml
trainer.max_train_steps: 10000
corrective_flow_scale: 0.2   # 降低 CF 损失比例，优先保证 base action 质量
trigger_threshold: 0.75      # 稍提高推理阈值，减少 H5 过修正
```

**注意**：在修复 H4/H5 之前，增加训练步数收益有限。

---

### 实验优先级建议

```
立即执行（无需新训练）:
  优先级 1: 方案 A（推理禁用消融）— 1.5 小时，确认根因
  优先级 2: 方案 B（阈值扫描）— 4 小时，量化 H5
  优先级 3: 方案 C（代码修复 + eval）— 2 小时，验证修复效果

如 C 后仍有 gap，再执行:
  优先级 4: 方案 D（增加训练步数）
```

---

## 方案可行性判定

### 建议：**PAUSE**（暂停，而非 PIVOT 或 CONTINUE）

**理由**：

1. **CF 方案本身有效性已经部分验证**：T0 +25pp, T6 +10pp 说明在正确的工作条件下 CF 确实有正收益。方案的核心假设（corrective flow 可以改善困难任务的成功率）是成立的。
2. **失败的根因是可修复的代码问题，而非方案缺陷**：H4（a_prev/a_base 不匹配）是一个明确可修复的推理路径设计缺陷，不需要重新训练。修复成本极低（约 30 行代码改动）。
3. **修复前无法有效判断 CF 的真实上限**：当前 -7pp 的结果掺杂了代码 bug 导致的退化，不能作为方案失败的最终判据。
4. **PIVOT 的条件尚未达到**：在尝试修复推理路径并重新测试之前，切换方案属于过早放弃。只有修复后仍然无法达到 -2pp 标准，才应考虑 PIVOT。

**暂停期间完成的动作**（按优先级）：

1. 执行方案 A（推理消融）确认根因
2. 如 A 确认推理路径是主要问题，执行方案 C（代码修复）
3. 对修复后 ckpt 重跑 CP-AH-4 eval
4. 如结果 >= Baseline - 2pp，CP-AH-4 PASS；否则进入方案 D 或重新评估

---

## 附录：证据链总结


| 假设                   | 支持证据                               | 最终判定                           |
| -------------------- | ---------------------------------- | ------------------------------ |
| H1 训练不足              | loss/cf_base=0.007≠0               | LIKELY 辅助因素                    |
| H2 损失干扰              | loss/action CF<Baseline            | UNLIKELY，排除                    |
| H3 trigger 质量        | loss/region 高方差                    | LIKELY 放大器                     |
| H4 a_prev/a_base 不匹配 | **代码审查发现，forward vs predict 语义不同** | **VERY LIKELY，首要根因**           |
| H5 过修正               | T1/T7/T8 一致 -10pp；region_gate≈0.5  | **VERY LIKELY，首要根因（T1/T7/T8）** |


*响应完成时间: 2026-04-15 CP-RES 线程*