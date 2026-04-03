# StarVLA 改造与统一伪标签生成任务书

> 适用范围更新（2026-04-03）：
> 当前仓库主线仅使用 `Qwen2.5VL / Qwen3VL`。`MapAnything/LLaVA3D/3D-VLM` 相关研发已解耦到其他 project，本任务书在本仓内不再将其作为必选实现路径。

## 文档定位

本文件是当前项目的**工程总任务书**，目标是在尽量不推翻现有训练框架的前提下，将现有的 StarVLA 模仿学习训练与推理框架改造成一个能够支持以下能力的统一研发底座：

1. 训练并输出 **base chunk policy**；
2. 离线回放轨迹并构建 **correction dataset**；
3. 支持 **A 模块（Dynamic Change Prior Module）** 的训练；
4. 支持 **Corrective Policy 模块** 的训练；
5. 为后续 **异步推理系统** 对接预留统一接口。

本任务书不讨论异步系统本身的实现细节，重点讨论：

- StarVLA 应如何改造；
- 统一数据流应该如何组织；
- 伪标签应该如何生成；
- A 与 corrective policy 如何共享样本生成器；
- 当前第一篇论文的最小可行研发路径是什么。

---

# 一、总目标与总体原则

## 1.1 当前阶段目标

当前阶段不是从零搭建一套新框架，而是：

> **在 StarVLA 的基础上做“最小侵入式改造”，使其成为一个支持 base chunk、伪标签生成、A 模块训练、corrective policy 训练的统一底座。**

## 1.2 为什么不从零重写

不从零搭建的原因包括：

1. 当前真正的科学问题不是框架工程，而是：
   - 动态变化表征是否真的有用；
   - correction 的对象应该是什么；
   - 剩余动作块的伪标签如何构造；
   - trigger / scope / delta chunk 如何设计。
2. 从零搭框架会消耗大量精力在数据接口、训练脚本、日志、分布式工程、checkpoint 管理上。
3. 第一篇论文更需要“主线闭合”，而不是“系统从零重写”。

## 1.3 总体原则

改造 StarVLA 时，必须坚持以下原则：

- **最小侵入式**：尽量保留现有 base policy 训练流程；
- **模块解耦**：base policy、伪标签生成、A 模块、corrective policy 四层分开；
- **统一数据真相来源**：A 和 corrective policy 使用同一套伪标签生成器；
- **先分阶段训练，再考虑联合微调**；
- **为异步系统预留接口，但不在当前阶段强耦合实现**。

---

# 二、StarVLA 需要被改造成什么

StarVLA 当前假设已经具备一个基础的模仿学习训练框架，可以完成 base policy 训练。当前需要将其改造成如下四层结构：

## 层 1：Base Chunk Policy 层
负责：

\[
(o_t, task) \rightarrow A_t^{base}
\]

其中：
- \(A_t^{base}\) 是长度为 \(H\) 的 nominal action chunk。

## 层 2：Correction Dataset Builder
负责对离线 demonstration 轨迹进行回放与伪标签生成，输出：

- stale remaining chunk
- teacher future chunk
- delta chunk target
- trigger label
- mismatch curve
- earliest mismatch index
- affected-region mask

## 层 3：A 模块训练器
使用统一伪标签生成器输出的样本，训练动态变化先验模块。

## 层 4：Corrective Policy 训练器
使用统一伪标签生成器输出的样本，以及 A 模块的输出，训练 corrective policy。

---

# 三、必须先审计 StarVLA 的哪些能力

在真正改造代码前，必须先检查 StarVLA 是否已经具备如下能力：

## 3.1 Base policy 是否支持 chunk 输出
需要确认：

- StarVLA 当前输出的是单步 action 还是 action chunk；
- 如果是单步 action，是否能够轻量改造为输出长度为 \(H\) 的动作块；
- chunk 的时间维是否能与 demonstration 对齐。

## 3.2 是否支持离线整轨迹回放
需要确认：

- 能否对一个 episode 中多个时刻 \(t_0\) 批量前向；
- 能否缓存 base chunk 输出；
- 能否离线回放而不依赖环境在线执行。

## 3.3 数据加载器是否支持时间窗口
需要确认：

- 是否能返回最近 \(K\) 帧 observation/state/action history；
- 是否方便按 episode 组织连续窗口，而不是只返回打散后的单步样本。

## 3.4 是否能方便增加新的 trainer
必须确认训练框架能否方便新增：

- `A_trainer`
- `corrective_policy_trainer`
- `dataset_builder`

如果不能，需要先做一次训练代码层的模块化清理。

---

# 四、统一数据流设计

这是整个改造中最重要的一部分。

## 4.1 原始数据统一格式

统一抽象为：

```python
Episode = {
    "obs": [o_0, o_1, ..., o_T],
    "state": [s_0, s_1, ..., s_T],
    "action": [a_0, a_1, ..., a_{T-1}],
    "task": task_text_or_id,
    "episode_id": ...
}
```

其中：
- `obs` 可以是 RGB、多视角图像、depth 等；
- `state` 包含 proprio / gripper / ee state；
- `action` 是 demonstration 动作序列；
- `task` 是文本指令或任务 ID。

## 4.2 基础设定

设：
- \(H\)：chunk 长度
- \(K\)：A 模块输入的历史窗口长度
- \(t_0\)：anchor 时刻
- \(k\)：已经执行的 nominal chunk 步数
- \(t=t_0+k\)：当前 correction 时刻

---

# 五、统一伪标签生成流程（核心）

这个流程必须写成一个独立脚本/模块，供 A 和 corrective policy 共用。

## Step 0：先有 base policy
先训练或固定 base StarVLA，使其可以输出：

\[
A_{t_0}^{base} = [a_{t_0}^{base}, ..., a_{t_0+H-1}^{base}]
\]

## Step 1：遍历 episode，选择 anchor 时刻 \(t_0\)
要求：
\[
t_0 + H \le T
\]

确保轨迹后面还有足够长度。

## Step 2：在 \(t_0\) 前向 base policy，生成 nominal chunk
输入：
- \(o_{t_0}\)
- \(s_{t_0}\)
- `task`

输出：
\[
A_{t_0}^{base}
\]

## Step 3：选择 offset \(k\)
表示 chunk 已经执行了 \(k\) 步。得到当前时刻：
\[
t = t_0 + k
\]

### k 的建议采样方式
第一版建议：
- 均匀采样 \(k \in [1, H-1]\)

之后可再引入更贴近 delay / execution horizon 的采样分布。

## Step 4：构造当前时刻的三类核心对象

### 4.1 当前 observation
\[
o_t, s_t
\]

### 4.2 当前 nominal remaining chunk
\[
A_t^{base,rem} = [a_{t_0+k}^{base}, ..., a_{t_0+H-1}^{base}]
\]

### 4.3 当前 ground-truth future chunk
从 demonstration 里取：
\[
A_t^{gt} = [a_t^{gt}, ..., a_{t+H_{rem}-1}^{gt}]
\]

其中：
\[
H_{rem}=H-k
\]

## Step 5：生成伪标签

### 5.1 Delta chunk target
\[
\Delta A_t^{target} = A_t^{gt} - A_t^{base,rem}
\]

### 5.2 Mismatch curve
逐位置比较：
\[
D_j = \|a_{t+j}^{gt} - a_{t+j}^{base,rem}\|
\]

### 5.3 Trigger label
\[
y_{trig} = \mathbb{1}[\max_j D_j > \epsilon_{trig}]
\]

### 5.4 Earliest mismatch index
\[
j^* = \min\{j \mid D_j > \epsilon_{scope}\}
\]

### 5.5 Affected-region mask
根据 \(j^*\) 构造：
- 前 \(j^*\) 个位置置 0
- 之后置 1

也可构造 soft mask。

### 5.6 Phase / time label（可选）
可由：
- 轨迹进度比例
- gripper 状态
- task script
- 接触事件

构造阶段标签。

---

# 六、统一样本格式设计

## 6.1 给 A 的样本格式

```python
A_sample = {
    "obs_window": [o_{t-K}, ..., o_t],
    "state_window": [s_{t-K}, ..., s_t],
    "action_window": [a_{t-K}, ..., a_{t-1}],
    "task": task,
    "future_change_target": y_future,
    "risk_target": y_risk,
    "region_target": y_region,
    "phase_target": y_phase,
}
```

## 6.2 给 corrective policy 的样本格式

```python
Corr_sample = {
    "obs_t": o_t,
    "state_t": s_t,
    "task": task,
    "base_chunk_full": A_{t_0}^{base},
    "remaining_chunk": A_t^{base,rem},
    "gt_future_chunk": A_t^{gt},
    "delta_chunk_target": Delta_A_t_target,
    "trigger_label": y_trig,
    "mismatch_curve": D,
    "earliest_mismatch_index": j_star,
    "affected_region_mask": m,
    "phase_or_time_index": tau_t,
    "history_window": [optional],
}
```

### 关键要求
A 和 corrective policy 的训练集必须来自同一个 sample builder，只是消费不同字段。

---

# 七、A 模块训练入口改造需求

StarVLA 中应新增一个 `A_trainer`，其职责是：

## 输入
\[
X_t^A = (o_{t-K:t}, s_{t-K:t}, a_{t-K:t-1})
\]

## 输出
- `dynamic_embedding`
- `risk_score`
- `affected_region_prior`（可选，推荐）
- `phase_score`（可选）

## 损失
建议第一版：
\[
\mathcal L_A = \lambda_1 \mathcal L_{future} + \lambda_2 \mathcal L_{risk}
\]

第二版加入：
\[
+ \lambda_3 \mathcal L_{region}
\]

### 注意
- 第一版不要求 end-to-end；
- A 模块训练完后先冻结；
- 输出接口必须稳定，供 corrective policy 消费。

---

# 八、Corrective Policy 训练入口改造需求

StarVLA 中应新增一个 `corrective_policy_trainer`，其职责是：

## 输入
\[
X_t^{corr} = (o_t, s_t, A_t^{base,rem}, z_t^{dyn}, r_t, m_t^{prior}, \tau_t)
\]

## 输出（第一版建议）
- `trigger` \(g_t\)
- `delta_chunk` \(\Delta A_t\)
- （可选）`refined_mask`

## 训练阶段划分

### 阶段 3B-1：warm-up
先不用 A 的真实输出，而用 oracle risk / oracle mask。

### 阶段 3B-2：transition
混合 oracle 先验和 A 的输出。

### 阶段 3B-3：final
只用 A 的真实输出训练与验证。

## 损失
第一版建议：
\[
\mathcal L_{corr} = \lambda_1 \mathcal L_{\Delta} + \lambda_2 \mathcal L_{trig} + \lambda_3 \mathcal L_{smooth}
\]

---

# 九、A 与 corrective policy 的正式模块协议

## 9.1 A 的输出协议

```python
A_output = {
    "dynamic_tokens": Tensor[Ntok, D],
    "dynamic_embedding": Tensor[D],
    "risk_score": Tensor[1] or Tensor[R],
    "affected_region_prior": Tensor[H_rem],  # optional but recommended
    "phase_score": optional Tensor[P],
}
```

## 9.2 Corrective policy 的输入协议

```python
Corr_input = {
    "obs_t": ...,
    "state_t": ...,
    "remaining_chunk": Tensor[H_rem, Da],
    "dynamic_embedding": Tensor[D],
    "risk_score": Tensor[1] or Tensor[R],
    "affected_region_prior": Tensor[H_rem],
    "phase_or_time_index": ...,
}
```

## 9.3 协议要求

1. A 的输出张量维度固定；
2. `affected_region_prior` 对齐当前 `remaining_chunk` 长度；
3. `risk_score` 的语义要统一（例如 correction need score）；
4. corrective policy 不能依赖部署期拿不到的 GT 标签。

---

# 十、代码层建议的模块组织

建议在 StarVLA 代码库中新增如下目录：

```text
starvla/
├── base_policy/
├── dataset_builder/
│   ├── build_correction_dataset.py
│   ├── pseudo_label_utils.py
│   └── sample_schema.py
├── modules/
│   ├── dynamic_prior_module.py
│   ├── corrective_policy_module.py
│   └── losses.py
├── trainers/
│   ├── train_base_policy.py
│   ├── train_dynamic_prior.py
│   └── train_corrective_policy.py
├── configs/
│   ├── base_policy.yaml
│   ├── dynamic_prior.yaml
│   └── corrective_policy.yaml
└── analysis/
    ├── visualize_risk.py
    ├── visualize_region_prior.py
    └── visualize_correction_examples.py
```

---

# 十一、分阶段训练方案（总流程）

## Phase 0：Base policy
- 训练 / 固定 StarVLA base chunk policy

## Phase 1：Dataset builder
- 生成统一 correction dataset

## Phase 2：Train A
- 先训 A 模块
- 产出：embedding + risk（+ region）
- 冻结

## Phase 3：Train corrective policy
### 3B-1 warm-up
- 输入：oracle risk / oracle mask

### 3B-2 transition
- 输入：oracle 与 A 输出混合

### 3B-3 final
- 输入：仅 A 输出

## Phase 4：Optional joint finetune
- 只联合 A 的轻量 head 与 corrective policy
- backbone 保持冻结

---

# 十二、指标与最小验证需求

虽然本任务书不负责完整实验设计，但在开发过程中必须至少记录：

## Base policy 层
- chunk rollout quality
- success（基础参考）

## A 模块层
- future prediction error
- risk prediction accuracy / AUC
- region prior quality（如果有）

## Corrective policy 层
- delta chunk reconstruction error
- trigger accuracy
- correction smoothness metrics
- 相比无 correction 的 improvement

---

# 十三、当前阶段建议的最小可行版本（MVP）

如果时间有限，建议先做以下最小版本：

## Base policy
- 先固定能输出 chunk 的 StarVLA

## A 模块
- 输入：历史窗口
- 输出：dynamic embedding + risk score
- 不急着做 affected-region prior

## Corrective policy
- 输入：observation + remaining chunk + dynamic embedding + risk + phase
- 输出：trigger + delta chunk
- 不急着做 refined mask

这样可以最快跑通主线。

---

# 十四、当前具体任务分解

## 任务 1：StarVLA 审计
输出：
- chunk 输出支持情况
- episode 回放支持情况
- 新 trainer 接入难度

## 任务 2：统一 sample builder
输出：
- 可保存的 correction dataset
- A_sample 与 Corr_sample 两种 schema

## 任务 3：A trainer
输出：
- dynamic embedding
- risk score

## 任务 4：corrective trainer
输出：
- trigger
- delta chunk

## 任务 5：联调接口
输出：
- A_output 与 Corr_input 对接成功

---

# 十五、对当前推进的最终建议

当前最该做的不是：
- 从零重写框架；
- 一开始就端到端联合训练；
- 一开始就加入 RL；
- 一开始就把异步 runtime 深度耦合进训练。

当前最该做的是：

> **先把 StarVLA 改造成一个统一底座，使其支持 base chunk、伪标签生成、A 模块训练和 corrective policy 训练。**

只有这一步完成，后续 A、corrective policy、async runtime 才能并行稳定推进。

---

# 十六、文档结语

StarVLA 当前最重要的使命，不是做最终系统，而是成为：

> **整个实时纠偏 VLA 项目的统一研发底座。**

只要这一步做对，后面：
- A 模块
- corrective policy
- async runtime
- 后续数据回流与后训练

都会有统一、稳定的数据与接口基础。
