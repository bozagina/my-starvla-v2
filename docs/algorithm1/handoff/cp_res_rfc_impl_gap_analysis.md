# CP-RES：RFC 意图 vs 当前实现差距分析

```
文档类型:   gap analysis
作者:       CP-RES 线程
日期:       2026-04-15
关联:       cp_res_policy_rfc.md, QwenPI.py, eval_libero.py, model2libero_interface.py
状态:       DELIVERED
```

---

## 背景

CP-AH-4 评测失败（CF 62% vs Baseline 69%，-7pp）触发本次差距梳理。  
本文件系统对比 `cp_res_policy_rfc.md` 的原始设计意图与当前代码实现，记录所有已确认的语义偏差和工程缺口，作为后续修复的依据。

---

## Gap 1：执行时序——同步预测式 vs 异步反应式（架构差距）

### RFC 原始意图

RFC §1.1 表述：

> "base policy（action chunk）**在执行过程中**会产生状态偏差 $d_k$，corrective policy 的目标是：**给定当前观测和 A-module 的评估输出**，预测何时、何处、以多大幅度对 base action 施加修正。"

"在执行过程中" 描述的是**反应式（reactive）**时序：

```
[执行 chunk_t，步骤 0-7]
         ↕ 同时（异步）
[A-module 评估状态偏差 d_k]
    └── trigger 触发 → CF head 修正 → 产出 chunk_{t+1}_corrected
[执行 chunk_{t+1}_corrected]
```

### 当前实现

`QwenPI.predict_action()`（L569-611）实现的是**预测式（predictive）**时序：

```
[predict_action() 调用]
  → 1. VLM forward（获取当前观测特征）
  → 2. base action DiT 预测（得到 chunk_t）
  → 3. A-module trigger 评估（用步骤 1 的特征）
  → 4. 若触发，CF head 修正 chunk_t
  → 5. 返回修正后的 chunk_t
[执行 chunk_t 的 8 步，期间模型不运行]
[predict_action() 再次调用]
```

A-module 评估和 CF 修正发生在 chunk 执行**之前**，使用执行前的观测，无法响应执行中产生的实际状态偏差。

### 对比

| 维度 | RFC 意图 | 当前实现 |
|---|---|---|
| 触发时机 | 执行中（状态偏差发生后） | 执行前（无法感知偏差） |
| CF 修正对象 | 即将执行的 chunk_{t+1} | 本次预测的 chunk_t（同轮） |
| 观测新鲜度 | 执行中的新观测 | chunk 开始前的观测 |
| 模式 | 反应式 | 预测式 |

### 修复方向

将 eval 客户端改造为**异步双线程**：

```
主线程（执行）:     执行 chunk_t → 从 action_queue 取动作
后台线程（推理）:   执行 chunk_t 期间 → predict_action(obs_current) → chunk_{t+1}
                      → 若 trigger 触发，CF 修正 chunk_{t+1}
                      → 放入 action_queue
```

参考：LeRobot RTC（`lerobot/policies/rtc/action_queue.py`）的 ActionQueue 设计模式。

**重要约束**：CF 修正粒度为 **chunk 级别（trigger 门控）**，而非 A2C2 风格的每步重算。trigger 触发时 CF head 只运行一次，修正整个 chunk_{t+1}。

**严重度**：HIGH  
**是否需要重训练**：否  
**修复成本**：中（eval 客户端工程改造，约 1-2 天）

---

## Gap 2：Episode Reset 链路断裂（工程缺口）

### RFC 意图

每个 episode 开始时，CF 的状态（`_cf_prev_chunk`）应该是干净的，不应携带上一个 episode 的状态。

### 已实现的部分（各层均存在）

```
QwenPI.reset_cf_state()              → ✅ 存在 (QwenPI.py L192-194)
WebsocketPolicyServer "reset" handler → ✅ 存在 (websocket_policy_server.py L143-146)
eval_libero.py client_model.reset()  → ✅ 每 episode 调用 (eval_libero.py L131)
```

### 缺失的一环

```
ModelClient.reset()         → ✗ 从不调用 self.client.reset()
WebsocketClientPolicy       → ✗ 没有 reset() 方法
```

`eval_libero.py` 调用 `client_model.reset()` 时，只重置了客户端侧的状态（`action_ensembler`、`image_history` 等），reset 信号永远无法到达 model server，`QwenPI.reset_cf_state()` 永远不会被触发。

### 实际后果

20 trials × 10 tasks = 200 个 episode，除第一个外，其余 190 个 episode 开始时 `_cf_prev_chunk` 携带上一个 episode 末尾的 chunk，导致每个 episode 前 8 步（第一个 chunk）的 CF 修正方向被污染。

### 修复方案

**Step 1**：`websocket_policy_client.py` 新增 `reset()` 方法：

```python
def reset(self) -> None:
    """通知 server 清空 episode 状态（如 _cf_prev_chunk）。"""
    data = self._packer.pack({"type": "reset"})
    self._ws.send(data)
    self._ws.recv()  # 等待 server 确认
```

**Step 2**：`model2libero_interface.py` 的 `ModelClient.reset()` 调用它：

```python
def reset(self, task_description: str) -> None:
    self.task_description = task_description
    self.client.reset()      # ← 新增：通知 server 清空 episode 状态
    self.image_history.clear()
    if self.action_ensemble:
        self.action_ensembler.reset()
    ...
```

**严重度**：MEDIUM  
**是否需要重训练**：否  
**修复成本**：低（约 10 行代码）

---

## Gap 3：Trigger 评估使用执行前的过期观测

### RFC 意图

trigger head 应检测"执行过程中产生的状态偏差 $d_k$"，即对**当前执行时刻**的状态进行评估。

### 当前实现

`predict_action()` L581-588：

```python
pooled_cf = base_hidden.float().mean(dim=1)   # base_hidden = chunk 开始前的 VLM 特征
trigger_prob = torch.sigmoid(ami.predict(pooled_hidden=pooled_cf, ...))
```

trigger 使用的 `base_hidden` 与 base action 预测共享同一次 VLM forward，即 chunk 执行前的观测。执行过程中产生的状态变化对 trigger 不可见。

### 与 Gap 1 的关系

Gap 1 修复（异步推理）后，trigger 评估仍使用缓存的 `base_hidden`（上一个 chunk 开始时的特征）。完整对齐 RFC 需要在 trigger 检查时使用新观测。

然而，根据设计决策：**CF 修正粒度为 chunk 级（trigger 门控），而非每步重算**。因此，trigger 使用缓存 `base_hidden` 的轻微延迟在工程上是可接受的折衷，不属于阻塞性问题。

若未来需要更精确的 trigger 响应，可考虑：
- 轻量级每步 state-based trigger（直接从 state observation 计算 $d_k$，不依赖 VLM）
- 但这属于 v1 方向，不在当前 v0 范围内

**严重度**：LOW（trigger 门控设计前提下可接受）  
**是否需要重训练**：否（v0 可接受）  
**修复成本**：依赖 Gap 1 修复后评估

---

## Gap 4：region_gate 收敛产物导致全 chunk 低幅度噪声

### RFC 意图

region 机制应只对"受影响区域"的 chunk 步骤施加修正，非受影响步骤的修正幅度应接近零。

### 训练结果

CF 33k ckpt 的 `loss/corrective_flow_region = 0.0003`，说明 region_logits 已收敛到接近零的值。

### 推理行为

`corrective_flow_head.py predict()` L221：

```python
region_gate = torch.sigmoid(region_logits).unsqueeze(-1)
# sigmoid(near_zero) ≈ 0.5 → 所有 8 步获得约 50% 幅度的速度修正
a_corrected = a_base + region_gate * pred_velocity
```

当 trigger 触发时，所有 8 个 chunk 步骤都被施加约 50% 幅度的速度修正，即使是"不需要修正"的步骤。

### 实际后果

对于 T1/T7/T8（Baseline 100%）这类已经完美执行的任务，trigger 偶尔误触发时，0.5 × velocity 的系统性噪声会破坏原本正确的轨迹，导致一致的 -10pp（从 100%→90%）。

### 修复方案

在 `corrective_flow_head.py` 的 `predict()` 方法中，对 region_gate 加阈值或偏移：

```python
# 方案 A：减去偏移，低于阈值的步骤完全不修正
region_gate = torch.clamp(
    torch.sigmoid(region_logits) - 0.3, min=0.0
).unsqueeze(-1)

# 方案 B：二值化阈值（更激进）
region_gate = (torch.sigmoid(region_logits) > 0.5).float().unsqueeze(-1)
```

**严重度**：MEDIUM（H5 过修正的直接根因）  
**是否需要重训练**：否  
**修复成本**：低（1 行代码改动，直接对 CF 33k ckpt 生效）

---

## Gap 5：a_prev 训练/推理分布轻微偏移

### 训练

`forward()` L473-477：

```python
a_prev = actions[:, :self.chunk_len, :]         # GT 前序 chunk（离线标注）
a_prev_noisy = a_prev + noise_scale * randn     # noise_scale=0.05
```

CF head 以 **GT 标注的前序 chunk**（加轻微噪声）为输入，学习预测修正速度。

### 推理

```python
a_prev_for_cf = self._cf_prev_chunk  # 上一次 predict_action 的模型输出
```

推理用的是**模型自身的前序预测**，而非 GT 标注。

### 偏差分析

在模型充分收敛的情况下，`_cf_prev_chunk ≈ GT 前序 chunk`，偏差可忽略。  
在 3000 步这种早期训练阶段，模型预测与 GT 之间有一定差距，CF head 看到的输入分布与训练时不同。

### 处置

非结构性问题，随训练步数增加自然收敛。当前阶段属于可接受的训练不足效应，不需要专项修复。

**严重度**：LOW  
**是否需要重训练**：否（增加训练步数可缓解）  
**修复成本**：自然收敛

---

## 差距总结

| # | 差距 | 类型 | 严重度 | 是否需要重训练 | 修复成本 |
|---|---|---|---|---|---|
| G1 | 执行时序：同步预测式 vs 异步反应式 | 架构差距 | **HIGH** | 否 | 中 |
| G2 | Episode reset 链路断裂 | 工程缺口 | MEDIUM | 否 | **低** |
| G3 | Trigger 使用执行前过期观测 | 语义偏差 | LOW | 否 | 依赖 G1 |
| G4 | region_gate ≈ 0.5 全覆盖噪声 | 收敛产物 | MEDIUM | 否 | **低** |
| G5 | a_prev 训练/推理分布偏移 | 轻微语义偏差 | LOW | 否（自然收敛） | 无 |

---

## 行动计划

### 阶段 1：即时修复（不需要重训练，不需要架构改动）

- [ ] **G2 修复**：`WebsocketClientPolicy.reset()` + `ModelClient.reset()` 调用链接通
- [ ] **G4 修复**：`corrective_flow_head.py` region_gate 阈值化
- [ ] 对 CF 33k ckpt 重跑 CP-AH-4 eval，验证修复效果

### 阶段 2：工程改造（不需要重训练）

- [ ] **G1 修复**：eval 客户端改造为异步双线程（执行/推理解耦）
  - 参考 LeRobot RTC 的 ActionQueue 模式
  - trigger 门控设计：CF 每个 chunk 最多运行一次（非每步重算）
- [ ] 重跑 CP-AH-4 eval，验证 G1 修复后的增量效果

### 阶段 3：长期 / 下一训练轮次

- [ ] **G5 缓解**：增加训练步数（建议 10k），使模型预测更接近 GT
- [ ] **G3 评估**：G1 修复后，评估 trigger 响应延迟是否成为瓶颈；若是，考虑 state-based trigger

---

*文档生成时间: 2026-04-15 CP-RES 线程*
