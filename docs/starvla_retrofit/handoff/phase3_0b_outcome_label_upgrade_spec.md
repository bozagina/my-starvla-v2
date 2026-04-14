# Phase3.0B Outcome Label Upgrade Specification

status: v0.9.3-draft
author: A-RES thread
round_id: A-ROUND-PHASE3_0B-OUTCOME-LABEL-UPGRADE
date: 2026-04-14
revision: |
  2026-04-14 v0.9   — initial spec with fixed α_d=0.35, τ_fail=0.7
  2026-04-14 v0.9.1 — auto-calibration amendment after LIBERO AH-2/AH-3 failure
  2026-04-14 v0.9.2 — training loss improvements: conditional region loss + consistency regularization
  2026-04-14 v0.9.3 — pseudo-label pipeline upgrade: intermediate states + progress signal + state-deviation-based correction_mask/region_prior

---

## 1. Problem Statement

当前 `build_fasa_dataset.py::_compute_pseudo_labels` 生成的伪标签完全基于相邻动作差分 `delta_t = a_{t+1} - a_t`。这导致三个结构性缺陷：

| 缺陷 | 根因（代码级） | 后果 |
|---|---|---|
| 风险定义错位 | `risk_score = max(delta_norm)` (L94) | 高动作变化 ≠ 高失败风险；低动作差但高失败风险的样本被漏判 |
| trigger 退化 | `trigger_label = max(correction_mask)` (L101)；`correction_mask = delta_norm >= Q75` (L93) | Q75 阈值使 ~25% 步数总被标记；只要 chunk 非平凡，trigger 几乎总为 1 |
| region 被污染 | `region_target_15 = max(prior, mask)` (L126) | 当 correction_mask 退化后 region 也趋向全激活，丧失定位能力 |

**目标**：将标签语义从"动作差分代理"升级为"未来结果定义"，同时保持字段名和消费接口不变。

---

## 2. Frozen Anchors（本轮不可变）

| 锚点 | 值 | 约束说明 |
|---|---|---|
| VLM | `qwen25_authoritative` | 不切换基线模型 |
| 七字段契约 | `risk_pred, trigger_logit, delta_pred, region_logits[15], dynamic_embedding[16], version, source` | 字段名和形状不变 |
| schema | `p1_shared_builder_v1` | 数据格式契约不变 |
| 训练消费接口 | `optional_loss_utils.py::build_optional_hook_targets()` | 字段名和 tensor shape 不变 |
| 模型结构 | `QwenPI.py` 中的 A-loss / corrective-loss heads | 不改模型结构 |

**改动边界**：只改构建脚本 (`tools/build_fasa_dataset.py`) 和审计脚本 (`tools/*audit*.py`)，不改模型和训练主逻辑。

---

## 3. Outcome Label Definitions（关闭 R1/R2/R3）

### 3.1 符号约定

| 符号 | 含义 |
|---|---|
| $t$ | 当前 anchor 时刻（sample_step） |
| $H$ | 未来观察窗口长度（默认 4，对齐 `--horizon-steps 4`） |
| $s_t$ | 当前状态向量 `state_t`，维度 $D_s$ |
| $s_{t+k}$ | 未来状态 `future_state`（当 $k=H$ 时已在 record 中） |
| $a_t^{base}$ | base policy 在 $t$ 时刻的 nominal action chunk |
| $D$ | action_dim（通常 7 = xyz + rotation + gripper） |

### 3.2 状态偏差 $d_k$（关闭 R2 和 R3）

**设计决策**：采用分组加权 L2 范数。

对状态向量的各维度分为三组，每组赋予不同权重：

| 组 | 维度索引（典型 7-DoF） | 权重 $w_g$ | 理由 |
|---|---|---|---|
| position (xyz) | `[0, 1, 2]` | 1.0 | 位置偏差是最直观的失败信号 |
| rotation | `[3, 4, 5]` | 0.5 | 旋转偏差影响较小，且角度环绕可能放大 L2 |
| gripper | `[6]` | 2.0 | 夹爪误差直接导致抓取/放置失败 |

加权偏差计算：

$$
d_k = \text{clip}\left(\frac{\sqrt{\sum_{j} w_{g(j)}^2 (s_{t+k}[j] - s_{t}[j])^2}}{\tau_s},\ 0,\ 1\right)
$$

**$\tau_s$ 标定规则**：

- 初始值：$\tau_s = 0.15$
- 标定方法：取训练集中全部 $(s_{t+H}, s_t)$ 对的加权 L2 偏差分布的 **P90** 值
- 如果 P90 < 0.05（数据几乎无偏差），则回退 $\tau_s = 0.05$ 并发出警告
- 如果 P90 > 0.5，则回退 $\tau_s = \text{P90}$ 并发出警告
- **BUILD 实现时必须在 stats 输出中报告 P90 和最终 $\tau_s$**

> **v0.9.1 补充说明**：$\tau_s$ = P90 normalization 意味着 d_H 分布被拉伸填满 [0, 1]（P90 处 d_H = 1.0，10% 饱和）。因此 $\alpha_d$ 和 $\tau_{fail}$ **不能**是固定常数——它们必须从 d_H 分布自适应标定。详见 §3.5。

**维度适配规则**（当 `state_dim != 7` 时）：

- 若 `state_dim >= 7`：前 3 维=position, 接 3 维=rotation, 第 7 维=gripper, 其余维权重=1.0
- 若 `state_dim < 7`：所有维度等权 $w=1.0$
- 权重向量记录在 stats 输出中以供审计

### 3.3 失败检测 $f_k$（关闭 R1）

**设计决策**：不依赖外部 failure oracle，用状态偏差阈值定义。

$$
f_k = \mathbb{1}[d_k > \tau_{fail}]
$$

| 参数 | v0.9 值 | v0.9.1 值 | 说明 |
|---|---|---|---|
| $\tau_{fail}$ | 0.7（固定） | **P(1 − target\_fail\_rate)(d\_H)，自适应** | 从 d_H 分布的高分位数自动标定 |
| target\_fail\_rate | （无） | **0.05（新参数）** | 控制 f_H=1 的目标比例 |

> **v0.9.1 修订**：$\tau_{fail} = 0.7$ 在 LIBERO 上导致 ~30–40% 样本 f_H=1，correction_mask 全激活比例过高，直接导致 AH-3 超标。改为自适应标定后，f_H=1 比例被精确控制在 target_fail_rate（默认 5%）。

理由：
1. 当前数据中没有显式 failure signal（如 success/fail 标注）
2. 自适应 $\tau_{fail}$ 确保 failure 检测只对分布中最极端的偏差触发
3. 此定义纯数据可计算，不需要额外标注
4. 未来如果有显式 success/fail 标注，可直接替换此定义而不改下游逻辑

**关键约束**：$f_k$ 只在 $k = H$（即 `future_state_step`）可计算。中间步 $k < H$ 因无中间状态而不可计算。

**处理方式**：
- 当仅有 $k=H$ 的 `future_state` 时：$f_k$ 只有一个值 $f_H$，此时所有公式中的 $\max_k f_k = f_H$
- 若 BUILD 后续实现多步 future_state 回填，公式自然扩展，无需改 spec

> **v0.9.3 实现**：中间步状态 $s_{t+1}, \ldots, s_{t+H}$ 已在 `_build_from_dataset_root` 中提取并存储为 `intermediate_states` 字段。逐步偏差 $d_k$ 用于 correction_mask 和 region_prior 计算（见 §3.4）。

### 3.4 Outcome Labels 计算公式

#### risk_score

$$
\text{risk\_score} = \text{clip}\left(\max\left(f_H,\ d_H\right),\ 0,\ 1\right)
$$

当仅有单步 future_state 时退化为 $\text{clip}(\max(f_H, d_H), 0, 1) = \text{clip}(d_H, 0, 1)$（因为 $f_H = \mathbb{1}[d_H > 0.7]$，若 $f_H=1$ 则 $d_H > 0.7$，$\max$ 取 1；若 $f_H=0$ 则取 $d_H$）。

#### trigger_label

$$
\text{trigger\_label} = \mathbb{1}\left[f_H = 1\ \text{or}\ d_H > \alpha_d\right]
$$

| 参数 | v0.9 值 | v0.9.1 值 | 说明 |
|---|---|---|---|
| $\alpha_d$ | 0.35（固定） | **P(1 − target\_trigger\_rate)(d\_H)，自适应** | 从 d_H 分布标定 |
| target\_trigger\_rate | （无） | **0.25（新参数）** | 控制 trigger=1 的目标比例 |

> **v0.9.1 修订**：$\alpha_d = 0.35$ 固定值在 LIBERO 上导致 82.9% trigger 正样率（远超 30% 门限）。根因：P90 归一化使 d_H 中位数 ≈ 0.66，远高于 0.35。改为 $\alpha_d = P_{75}(d_H)$（对应 25% 目标触发率）后，LIBERO 上 $\alpha_d \approx 0.86$，trigger 率 ≈ 25%。

#### delta_action_norm（保留兼容）

保留原始 `delta_norm` 计算作为辅助信号，但增加 outcome 分量：

$$
\text{delta\_action\_norm} = \max\left(\max_j \|\Delta a_j\|,\ \beta \cdot d_H\right)
$$

| 参数 | 值 | 说明 |
|---|---|---|
| $\beta$ | 0.5 | 将状态偏差映射回动作空间的缩放系数 |

理由：纯状态偏差不能直接等价于动作纠偏量，$\beta=0.5$ 作为保守混合，保留原始 delta 信号的同时引入 outcome 信号。

> **v0.9.2 实现说明**：`delta_action_norm` 字段存储的是上述 outcome-mixed scalar 值（非 per-step list）。原始 per-step delta norm list 保留在 `_outcome_v2.delta_norm_per_step` 中供审计使用。下游 `optional_loss_utils._coerce_delta_norm` 直接消费此 scalar，无需 max 归约。

#### affected_region_prior（用于构建 region_target_15）

对 $H=4$ 窗口，将 outcome 信号叠加到 15 个 bin 的后半段（因为偏差在 chunk 尾部更明显）：

$$
\text{region\_prior}[i] = \begin{cases}
\text{clip}(\delta\_norm\_prior[i] + \gamma \cdot \max(0,\ d_H - \alpha_d), 0, 1) & \text{if } i \geq \lfloor 15/2 \rfloor \\
\text{clip}(\delta\_norm\_prior[i], 0, 1) & \text{otherwise}
\end{cases}
$$

其中 $\delta\_norm\_prior$ 是原始动作差分归一化分布，$\gamma = 0.3$。

> **v0.9.1 修订**：将 $\gamma \cdot d_H$ 改为 $\gamma \cdot \max(0, d_H - \alpha_d)$。原始公式对所有样本无差别叠加 $\gamma \times d_H$，在 d_H 中位数 ≈ 0.66 时导致后半段 bin 普遍被推高 ~0.2，使 AH-3 超标。新公式仅在 d_H 超过 trigger 阈值 $\alpha_d$ 时叠加**超额部分**，非触发样本的 region prior 不受 outcome 信号干扰。

#### correction_mask

v0.9.1（legacy）：

$$
\text{correction\_mask}[j] = \mathbb{1}\left[\delta\_norm[j] \geq Q_{75}(\delta\_norm)\ \text{or}\ f_H = 1\right]
$$

> **v0.9.3 升级**：消除动作差分依赖，改为基于逐步状态偏差：
>
> $$
> d_k = \text{clip}\left(\frac{\|s_{t+k} - s_t\|_W}{\tau_s},\ 0,\ 1\right), \quad k = 1, \ldots, H
> $$
>
> $$
> \text{correction\_mask}[k] = \mathbb{1}\left[d_k > \alpha_d\ \text{or}\ f_H = 1\right]
> $$
>
> 当 $f_H = 1$ 时整个 mask 激活为全 1（与 v0.9.1 行为一致）。
> correction_mask 和 region_prior 长度从 `action_chunk_len - 1` 变为 `horizon_steps`，下游 `_build_region_target_15` 重采样到 15 bin 不受影响。

#### affected_region_prior（v0.9.3 升级）

> **v0.9.3**：region_prior 基底从 `delta_norm_prior`（动作差分归一化分布）改为 `d_k` 归一化分布：
>
> $$
> d\_k\_prior[k] = \frac{d_k}{\sum_{k=1}^{H} d_k}
> $$
>
> outcome 叠加公式不变（$\gamma \cdot \max(0, d_H - \alpha_d)$ 叠加到后半段）。

#### region_target_15（最终合成）

保持原始合成逻辑 `max(prior, mask)` 不变，但因为上游 `prior` 和 `mask` 的语义已升级（v0.9.3 基于状态偏差），下游自动受益。

### 3.5 自适应超参数标定（v0.9.1 新增）

> 本节替代 v0.9 中 $\alpha_d = 0.35$（固定）和 $\tau_{fail} = 0.7$（固定）的设定。

#### 背景

v0.9 在 LIBERO 2000-sample 数据上的实测结果：

| 指标 | 实测值 | 门限 | 状态 |
|---|---|---|---|
| AH-1 零差分误触发率 | 0% | < 1% | PASS |
| AH-2 trigger 正样率 | 82.9% | 1%–30% | **FAIL (2.76×)** |
| AH-3 region 平均激活 bin | 7.91/15 | < 5 | **FAIL (1.58×)** |

根因：$\tau_s$ = P90 = 0.0678 将 d_H 分布拉伸至 [0, 1]（中位数 ≈ 0.66），而 $\alpha_d = 0.35$ 和 $\tau_{fail} = 0.7$ 作为固定绝对阈值无法适配数据驱动的 d_H 分布。

#### 标定流程（扩展 two-pass 框架）

在现有 two-pass 框架的 Pass 1 中增加标定步骤：

```
Pass 1:
  1. 收集所有 raw_deviation（已有）
  2. τ_s = clamp(P90(raw_deviations), 0.05, 0.5)（已有）
  3. 计算 d_H_all = [clip(r / τ_s, 0, 1) for r in raw_deviations]（新增）
  4. α_d = quantile(d_H_all, 1.0 - target_trigger_rate)（新增）
  5. τ_fail = quantile(d_H_all, 1.0 - target_fail_rate)（新增）

Pass 2:
  用 (τ_s, α_d, τ_fail) 计算所有标签（已有流程，参数来源变更）
```

#### 新增 CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--target-trigger-rate` | 0.25 | trigger=1 的目标比例（α_d 从此推导） |
| `--target-fail-rate` | 0.05 | f_H=1 的目标比例（τ_fail 从此推导） |

#### 数学约束

AH-3 对 target_fail_rate 的硬约束（设 normal_bins ≈ 3.5）：

$$
p_f \times 15 + (1 - p_f) \times 3.5 < 5 \implies p_f < 0.13
$$

因此 target_fail_rate 必须 < 13%。默认 5% 留有安全余量。

#### stats 输出要求

标定结果必须在 stats JSON 中完整报告：

```json
{
  "calibration": {
    "calibrated_tau_s": ...,
    "calibrated_alpha_d": ...,
    "calibrated_tau_fail": ...,
    "target_trigger_rate": 0.25,
    "target_fail_rate": 0.05,
    "d_H_P10": ..., "d_H_P25": ..., "d_H_P50": ...,
    "d_H_P75": ..., "d_H_P90": ..., "d_H_P95": ...,
    "d_H_max": ...,
    "raw_deviation_P90": ...
  }
}
```

#### AH-1 安全性保证

零差分样本（raw_deviation = 0）→ d_H = 0 → d_H < α_d（因为 α_d > 0）→ trigger = 0。AH-1 恒等通过，不受自适应参数影响。

#### LIBERO 预期效果

| 参数 | v0.9 固定值 | v0.9.1 LIBERO 预估值 |
|---|---|---|
| τ_s | 0.0678 (P90) | 0.0678 (P90, 不变) |
| α_d | 0.35 | ~0.86 (P75 of d_H) |
| τ_fail | 0.7 | ~0.97 (P95 of d_H) |
| trigger 正样率 | 82.9% | ~25% |
| f_H=1 比例 | ~35% | ~5% |
| region 平均激活 bin | 7.91 | ~4.2 |

---

### 3.6 v0.9.3 伪标签管线升级（新增）

#### 新增 record 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `intermediate_states` | `list[list[float]]` | 中间步状态 $s_{t+1}, \ldots, s_{t+H}$，长度 = horizon_steps。`intermediate_states[-1]` = `future_state`（向后兼容）。 |
| `progress_t` | `float \| null` | 轨迹进度信号 $p_t = \text{frame\_index} / \max(1, \text{episode\_length} - 1)$，范围 [0, 1]。correction_jsonl 来源若无进度信息则为 null。 |
| `meta.episode_length` | `int` | 所属 episode 的帧数（仅 dataset-root 模式）。 |

#### 标签变更

| 字段 | v0.9.1 | v0.9.3 |
|---|---|---|
| `correction_mask` | 基于 $\delta\_norm[j] \geq Q_{75}$ 或 $f_H=1$；长度 = action_chunk_len - 1 | 基于 $d_k > \alpha_d$ 或 $f_H=1$；长度 = horizon_steps |
| `affected_region_prior` | 基底 = delta_norm 归一化分布 | 基底 = $d_k$ 归一化分布 |
| `trigger_label`, `risk_score`, `delta_action_norm` | 不变 | 不变 |
| `_outcome_v3` | （不存在） | 包含 `d_k` 列表、`d_H`、`f_H` 等诊断信息 |

#### 新增函数

- `_compute_pseudo_labels_v3()`：接受 `intermediate_states` 参数，使用逐步状态偏差 $d_k$ 替代动作差分。
- `_canonicalize_pseudo_labels()` 自动路由：有 intermediate_states 时用 v3，否则回退 v2。

#### `--legacy-labels` 兼容

legacy 模式不受影响：correction_mask 和 region_prior 保持 action_chunk_len - 1 长度、delta_norm 基底。新字段（intermediate_states、progress_t）仍然被添加到 record 中（加法性兼容）。

---

## 4. Implementation Scope for A-BUILD

### 4.1 必须修改的文件

| 文件 | 改动 |
|---|---|
| `tools/build_fasa_dataset.py` | (1) 将 `_calibrate_tau_s()` 扩展为 `_calibrate_outcome_params()` 以同时标定 τ_s、α_d、τ_fail (§3.5)；(2) 修改 `_compute_pseudo_labels_v2()` 中 γ_region 公式为 `γ * max(0, d_H - α_d)` (§3.4)；(3) 新增 `--target-trigger-rate` 和 `--target-fail-rate` CLI 参数；(4) stats 输出中报告完整 d_H 分位数和标定后参数 |
| `tools/fasa_dataset_audit.py` | 增加 outcome label 验收指标的检查（trigger 正样率、region 激活 bin 数） |

### 4.2 可选修改的文件

| 文件 | 改动 | 条件 |
|---|---|---|
| `tools/build_fasa_a_outputs.py` | 若 surrogate 需同步升级 | 由 A-BUILD 评估后决定 |
| `tools/a_outputs_audit.py` | 增加 outcome label 审计 | 可延后到下一轮 |

### 4.3 不得修改的文件

| 文件 | 冻结范围 | 说明 |
|---|---|---|
| `starVLA/model/framework/QwenPI.py` | 模型结构（heads、forward 签名） | **Carve-out**：`_compute_optional_hook_outputs` 中的 loss 计算逻辑允许修改（§4.4 P0+P1），但不得新增/删除/重命名 heads，不得改变 forward 输入输出接口 |
| `starVLA/model/framework/a_module_interface.py` | 全文件冻结 | 不可修改 |
| `starVLA/model/framework/optional_loss_utils.py` | 全文件冻结 | 不可修改 |
| `starVLA/training/**` | 超出 RES 改动范围 | A-BUILD 可修改 metric 报告逻辑（如 `decomposed_metric_specs`），不得改变训练主流程 |

### 4.4 Training Loss Improvements（v0.9.2 新增）

> 本节定义与 Outcome Labels v0.9.1 配套的训练损失改进。改进分两个优先级实施。

#### 背景

当前 QwenPI.py 中的 A-loss 和 corrective-loss 存在以下结构性问题：

| 问题 | 根因 | 后果 |
|---|---|---|
| region loss 在 trigger=0 样本上浪费梯度 | `region_prior_mask` 不区分 trigger 状态 | trigger=0 的样本 region prior ≈ 原始 delta 分布（无 outcome 信号），模型被迫拟合噪声 |
| trigger 类别不平衡 | v0.9.1 目标 trigger 率 25% → 正负样本 1:3 | BCE 被负样本主导，trigger=1 的 recall 受损 |
| risk/trigger/region 头独立训练 | 三个 loss 各算各的，无跨头约束 | 可能出现 trigger=0 但 region 高激活（语义矛盾），或 risk 高但 trigger 低（不一致） |

#### P0：Region Loss Conditional on Trigger（高优先级）

**核心变更**：`corrective_loss_region`（包含 `correction_mask` BCE 和 `region_prior` MSE）仅在 `trigger_label=1` 的样本上计算。

**实施方式**：在 `QwenPI.py` 的 corrective_loss 计算中，将 `correction_mask_mask` 和 `region_prior_mask` 与 `trigger_label > 0.5` 做交集：

```python
# 在 corrective_loss 块中，region 相关 loss 仅对 trigger=1 样本计算
trigger_positive = targets["trigger_label"] > 0.5  # [B]

if targets["correction_mask_mask"].any():
    correction_mask = targets["correction_mask_mask"] & trigger_positive
    if correction_mask.any():
        region_component = region_component + correction_mask_weight * F.binary_cross_entropy_with_logits(
            region_logits[correction_mask],
            targets["correction_mask"][correction_mask].clamp(0.0, 1.0),
        )
        correction_mask_target_count += int(correction_mask.sum().item())

if targets["region_prior_mask"].any():
    region_mask = targets["region_prior_mask"] & trigger_positive
    if region_mask.any():
        region_prob = torch.sigmoid(region_logits[region_mask])
        region_component = region_component + region_prior_weight * F.mse_loss(
            region_prob,
            targets["region_prior"][region_mask].clamp(0.0, 1.0),
        )
        region_prior_target_count += int(region_mask.sum().item())
```

**理由**：

1. trigger=0 意味着 A 模块判断不需要纠偏，此时 region 的语义是"无受影响区域"。强迫模型拟合 trigger=0 样本的 region prior（本质是动作差分噪声）会引入虚假监督信号。
2. 条件化后，region head 只从真正需要纠偏的样本中学习，监督信号更纯净。
3. 当 target_trigger_rate=0.25 时，region 有效训练样本减少到 25%，但这些样本的 region prior 具有真实 outcome 语义——质量远优于数量。

**预期效果**：

- region loss 下降更快（因为去除了噪声标签）
- trigger=0 样本不再产生 region 梯度（减少梯度冲突）
- 最终 region 预测在 trigger=1 时更精准，trigger=0 时 logits 自然趋向低值

**兼容性**：纯代码改动（QwenPI.py），不改伪标签格式或字段名。

#### P1：Trigger-Region 一致性正则（中优先级）

**核心变更**：在 a_loss 中新增一项一致性正则 $\mathcal{L}_{consist}$，惩罚 trigger 预测与 region 激活强度之间的语义矛盾。

**公式**：

$$
\mathcal{L}_{consist} = \frac{1}{B} \sum_{i=1}^{B} \max\left(0,\ \bar{p}_{region}^{(i)} - \sigma(\text{trigger\_logit}^{(i)}) - \epsilon\right)^2
$$

其中 $\bar{p}_{region}^{(i)} = \frac{1}{15}\sum_{j=1}^{15} \sigma(\text{region\_logits}^{(i)}_j)$ 是第 $i$ 个样本的 region 平均激活概率，$\sigma$ 是 sigmoid，$\epsilon = 0.1$ 是容忍余量。

**语义**：当 region 平均激活概率显著高于 trigger 概率时（即"region 认为有很多受影响区域，但 trigger 认为不需要纠偏"），施加二次惩罚。反向不惩罚——trigger 高但 region 低是合法的（如小范围高风险纠偏）。

| 参数 | 符号 | 默认值 | CLI 参数 | 说明 |
|---|---|---|---|---|
| 一致性正则权重 | $\lambda_{consist}$ | 0.1 | `consist_weight` in config | 初始较小，避免过度约束 |
| 容忍余量 | $\epsilon$ | 0.1 | 内置 | 允许 region 略高于 trigger |

**实施位置**：`QwenPI.py` 的 a_loss 或 corrective_loss 块尾部，作为额外正则项。

```python
# consistency regularization (P1)
consist_weight = float(_cfg_get(a_cfg, "consist_weight", 0.1))
if consist_weight > 0 and region_logits is not None:
    trigger_prob = torch.sigmoid(trigger_logit)          # [B]
    region_prob_mean = torch.sigmoid(region_logits).mean(dim=-1)  # [B]
    margin = region_prob_mean - trigger_prob - 0.1
    consist_loss = consist_weight * (F.relu(margin) ** 2).mean()
    a_loss = a_loss + consist_loss
    output_dict["a_loss_consist"] = consist_loss
```

**理由**：

1. 当前 trigger head 和 region head 完全独立训练，可能产生语义矛盾的预测。
2. 一致性正则是 soft constraint，不硬编码 trigger 和 region 的关系，而是对最不合理的矛盾施加惩罚。
3. 单向设计：只惩罚"region 高 + trigger 低"，不惩罚"trigger 高 + region 低"，因为后者语义合法（全局风险但无法精确定位）。

**预期效果**：

- 减少推理时出现"trigger=0 但 region 大面积激活"的矛盾输出
- 间接提升 trigger 的 recall（region 高激活会"拉高" trigger 概率）
- 对整体 loss 量级影响小（$\lambda_{consist}=0.1$ + hinge margin）

**兼容性**：纯代码改动（QwenPI.py），不改伪标签格式。新增一个 config 参数 `consist_weight`。

#### P0/P1 实施顺序

| 步骤 | 内容 | 改动文件 | 验证 |
|---|---|---|---|
| 1 | 实施 P0（conditional region loss） | `QwenPI.py` | 500-step 训练，确认 region loss 下降更快、无 NaN |
| 2 | 实施 P1（consistency regularization） | `QwenPI.py` | 500-step 训练，确认 `a_loss_consist` 有限且逐步下降 |
| 3 | P0+P1 联合验证 | — | AH-4 全通过（所有 loss 有限且非退化） |

---

## 5. Acceptance Hypothesis

> **如果 Outcome Labels v0.9 正确实现，以下五条可测量条件应全部满足**：

| # | 指标 | 门限 | 测量方式 | 对应旧标签问题 |
|---|---|---|---|---|
| AH-1 | 零差分样本误触发率 | < 1% | 筛选 `max(delta_norm) < 1e-6` 的样本，统计 `trigger_label=1` 的比例 | trigger 退化 |
| AH-2 | `trigger_label` 正样率 | 1% ~ 30% | 全数据集统计（当前接近 100%） | trigger 退化 |
| AH-3 | `region_target_15` 平均激活 bin 数 | < 5/15 | 全数据集统计 $\text{mean}(\sum_i \mathbb{1}[\text{region}[i] > 0.5])$ | region 污染 |
| AH-4 | 500-step 所有分解 loss 有限且非退化 | 全通过 | 训练 `metrics.jsonl` 中 `a_loss_*`（含 `a_loss_consist`）和 `corrective_loss_*` 全为有限值 | 功能性验证 |
| AH-5 | `Success@LIBERO` 不劣于当前基线 | > -2pp | 远程评估 | 端到端验收 |

AH-1 ~ AH-3 可在 BUILD 阶段本地验证；AH-4 需要训练 500 步；AH-5 需要远程评估。

---

## 6. Open Risks（降级后剩余风险）

| # | 风险 | 严重度 | 缓解措施 |
|---|---|---|---|
| R1' | $f_k$ 仅基于状态偏差阈值，可能遗漏状态偏差小但语义失败的场景（如物体滑落但夹爪位置未变） | LOW | 夹爪维度权重=2.0 部分缓解；未来可用显式 success/fail 标注替换 |
| R2' | 自适应标定依赖训练集分布，泛化到新数据集时 α_d/τ_fail 可能不适用 | LOW | 标定结果全部记录在 stats 中；自适应机制天然适配新分布 |
| R4 | spec v0.9.2 修订后尚未经 A-REVIEW 审核 | MEDIUM | 本轮产出后交 A-REVIEW |
| R5 | 当前 trigger 退化率已有 LIBERO 量化数据（82.9%），但仅一个数据集 | LOW | A-BUILD 用推荐参数重跑后对比验证 |
| R6 | 双峰分布数据集（混合任务）中 percentile 标定可能不是最优分割 | LOW | 审计时检查 d_H histogram，必要时分任务标定 |
| R10 | P0 conditional region loss 使 region 有效训练样本降至 ~25%，可能导致 region head 收敛变慢 | LOW | 监控 `corrective_loss_region` 收敛曲线；trigger_rate 可调高至 0.35 增加样本量 |
| R11 | P1 一致性正则 $\lambda_{consist}$ 过大时可能压制 region 多样性 | LOW | 默认 0.1 保守；训练中监控 `a_loss_consist` 量级 |

R1（$f_k$ 定义）、R2（$\tau_s$ 标定）、R3（$W$ 矩阵）已在 §3 中关闭。
R7（α_d 固定值不适配）、R8（τ_fail 固定值不适配）、R9（γ_region 无条件叠加过度激活）已在 §3.5 中关闭。

---

## 7. Parameter Summary

### 7.1 直接参数

| 参数 | 符号 | 默认值 | CLI 参数 | 可调范围 | v0.9.1 变更 |
|---|---|---|---|---|---|
| 未来窗口 | $H$ | 4 | `--horizon-steps` (已有) | 2~8 | 无 |
| 状态偏差归一化 | $\tau_s$ | 0.15 (P90 标定覆盖) | `--tau-s` | 0.05~0.5 | 无 |
| 失败阈值 | $\tau_{fail}$ | ~~0.7~~ → **auto** | `--tau-fail` | auto 或手动 | **自适应标定** |
| trigger 阈值 | $\alpha_d$ | ~~0.35~~ → **auto** | `--alpha-d` | auto 或手动 | **自适应标定** |
| delta 混合系数 | $\beta$ | 0.5 | `--beta-delta` | 0.0~1.0 | 无 |
| region 混合系数 | $\gamma$ | 0.3 | `--gamma-region` | 0.0~0.5 | **公式变更** |
| position 权重 | $w_{pos}$ | 1.0 | 内置 | -- | 无 |
| rotation 权重 | $w_{rot}$ | 0.5 | 内置 | -- | 无 |
| gripper 权重 | $w_{grip}$ | 2.0 | 内置 | -- | 无 |

### 7.2 自适应标定控制参数（v0.9.1 新增）

| 参数 | 默认值 | CLI 参数 | 安全调节区间 | 说明 |
|---|---|---|---|---|
| target_trigger_rate | 0.25 | `--target-trigger-rate` | 0.15 ~ 0.35 | <0.15 过保守漏报；>0.35 AH-2 超标风险 |
| target_fail_rate | 0.05 | `--target-fail-rate` | 0.02 ~ 0.10 | >0.13 AH-3 必然超标 |

### 7.3 训练损失参数（v0.9.2 新增）

| 参数 | 符号 | 默认值 | 配置位置 | 说明 |
|---|---|---|---|---|
| 一致性正则权重 | $\lambda_{consist}$ | 0.1 | `a_loss.consist_weight` | P1：过大压制 region 多样性 |
| 一致性容忍余量 | $\epsilon$ | 0.1 | 内置 | 允许 region 略高于 trigger |

### 7.4 公式变更摘要

| 公式 | v0.9 | v0.9.1 | v0.9.2 |
|---|---|---|---|
| α_d | 0.35（固定常数） | P(1 − target_trigger_rate)(d_H)（自适应） | （同 v0.9.1） |
| τ_fail | 0.7（固定常数） | P(1 − target_fail_rate)(d_H)（自适应） | （同 v0.9.1） |
| γ_region 叠加 | γ × d_H（无条件） | γ × max(0, d_H − α_d)（仅超额部分） | （同 v0.9.1） |
| region loss scope | 所有样本 | 所有样本 | **仅 trigger=1 样本**（P0） |
| trigger-region 约束 | 无 | 无 | **一致性正则 $\mathcal{L}_{consist}$**（P1） |

---

## 8. Handoff Checklist

### For A-BUILD

#### 伪标签构建（v0.9 → v0.9.1）

- [x] ~~实现 `_compute_pseudo_labels_v2()` 替代旧函数，遵循 §3.4 公式~~（v0.9 已完成）
- [x] ~~增加 `--tau-s`, `--tau-fail`, `--alpha-d` 命令行参数~~（v0.9 已完成）
- [x] ~~将 `_calibrate_tau_s()` 扩展为 `_calibrate_outcome_params()`~~（v0.9.1 已完成）
- [x] ~~增加 `--target-trigger-rate` / `--target-fail-rate` CLI 参数~~（v0.9.1 已完成）
- [x] ~~修改 γ_region 公式为 conditional~~（v0.9.1 已完成）
- [x] ~~LIBERO AH-1~AH-3 全部通过~~（v0.9.1 远程已验证）

#### 训练损失改进（v0.9.2）

- [ ] **P0**：修改 `QwenPI.py` corrective_loss 块，region 相关 loss 仅对 trigger=1 样本计算（§4.4 P0）
- [ ] **P1**：在 `QwenPI.py` a_loss 块尾部添加 trigger-region 一致性正则（§4.4 P1）
- [ ] 在 config 中新增 `consist_weight` 参数（默认 0.1）
- [ ] 500-step 训练验证 AH-4：所有 loss（含 `a_loss_consist`）有限且非退化
- [ ] 新增 debug metrics：`a_loss_consist`、`debug/corrective_loss_region_trigger_filtered_count`
- [ ] 将 changed files 和 evidence 交 A-REVIEW

### For A-REVIEW

- [ ] 审核本 spec v0.9.2 中的公式和参数是否自洽（§3.5 自适应标定 + §4.4 训练损失改进）
- [ ] 确认冻结锚点未被违反（特别是 QwenPI.py 改动不影响模型结构/字段名）
- [ ] 确认 AH-1 ~ AH-5 的测量方式可复现
- [ ] 验证 P0 实施正确性：trigger=0 样本不产生 region 梯度
- [ ] 验证 P1 公式正确性：单向 hinge margin，不惩罚 "trigger 高 + region 低"
- [ ] 对 BUILD 产出物做 findings list + pass/fail
