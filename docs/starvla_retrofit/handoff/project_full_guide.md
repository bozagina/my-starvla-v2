# StarVLA 实时纠偏系统：项目完整指南

最后更新：2026-04-14（Asia/Shanghai）
适用读者：第一次接触本项目、需要从零理解整体架构、数据流、训练范式的研发同学

---

## 一、项目一句话定位

> 在 StarVLA 模仿学习框架的基础上，做最小侵入式改造，使其成为一个支持 **base chunk policy → 统一伪标签生成 → A 模块训练 → corrective policy 训练 → 实时执行纠偏** 的统一研发底座。

当前 VLM 范围：仅 `Qwen2.5-VL` / `Qwen3-VL`。

---

## 二、你需要知道的核心概念

### 2.1 Base Chunk Policy

机器人接到任务指令和视觉观测后，输出一个长度为 $H$ 的动作序列（action chunk）：

$$
(o_t, \text{task}) \xrightarrow{\text{base policy}} A_t^{base} = [a_t, a_{t+1}, \ldots, a_{t+H-1}]
$$

这是 StarVLA 原始训练框架就有的能力。当前使用的框架是 **Qwen-PI**（Flow-Matching 扩散式动作预测）。

### 2.2 为什么需要纠偏

Base policy 输出的 action chunk 是一个"规划"，但在开环执行过程中，环境可能发生变化（物体位移、接触力变化、任务阶段切换等），导致 chunk 后半段与实际需求产生偏差。

核心问题：**在 chunk 执行到第 $k$ 步时，如何判断"是否需要纠偏"以及"怎么纠偏"**。

### 2.3 两个核心模块


| 模块                    | 全称                          | 职责                     | 输入                   | 输出                      |
| --------------------- | --------------------------- | ---------------------- | -------------------- | ----------------------- |
| **A 模块**              | Dynamic Change Prior Module | 判断"当前环境是否发生了需要纠偏的变化"   | 最近观测窗口、状态窗口、动作历史     | 风险分数、触发信号、受影响区域、动态嵌入    |
| **Corrective Policy** | Residual/Correction Policy  | 在 A 模块判断需要纠偏时，输出"纠偏动作" | 当前观测、剩余 chunk、A 模块输出 | delta action chunk（纠偏量） |


两者的关系：**A 模块是 corrective policy 的上游**，corrective policy 消费 A 模块的输出来决定如何纠偏。

> 在本仓库中，`corrective policy` 与 `residual policy` 语义等价。

---

## 三、系统全景架构

```text
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段（离线）                          │
│                                                                 │
│  Phase 0   Phase 1           Phase 2         Phase 3            │
│  ───────   ──────────────    ──────────      ──────────         │
│  Base      统一伪标签生成    A 模块训练      Corrective         │
│  Policy    (Correction       (冻结后提供      Policy 训练       │
│  训练       Dataset Builder)  输出给下游)      (消费A输出)       │
│                                                                 │
│  输出:      输出:              输出:           输出:             │
│  base       pseudo_labels     risk_pred       trigger           │
│  chunk      a_outputs         trigger_logit   delta_action      │
│  policy     region_target     region_logits                     │
│                               dynamic_embed                     │
└─────────────────────────────────────────────────────────────────┘

                            ↓ 部署时

┌─────────────────────────────────────────────────────────────────┐
│                       推理阶段（在线）                           │
│                                                                 │
│  观测 o_t ──→ Base Policy ──→ A_t^base (action chunk)          │
│                                                                 │
│  执行到第 k 步时:                                                │
│  (o_t, s_t, 执行视频) ──→ A 模块 ──→ {risk, trigger, region}   │
│                                           │                     │
│                    trigger=1?  ───yes───→  Corrective Policy    │
│                        │                   → delta_action       │
│                       no                   → 修正剩余 chunk      │
│                        │                                        │
│                     继续执行原 chunk                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、数据流详解

### 4.1 原始数据格式

所有数据统一抽象为 episode 结构：

```python
Episode = {
    "obs":    [o_0, o_1, ..., o_T],       # 视觉观测序列（RGB / 多视角）
    "state":  [s_0, s_1, ..., s_T],       # 本体状态（关节角度 / 末端位姿 / 夹爪）
    "action": [a_0, a_1, ..., a_{T-1}],   # 示教动作序列
    "task":   "把杯子放到盘子上",           # 任务指令
}
```

### 4.2 统一伪标签生成（Phase 1）

这是整个系统最关键的数据构建步骤。A 模块和 corrective policy 共享同一个 sample builder。

**核心流程**：

1. 在 episode 中选择 anchor 时刻 $t_0$
2. 用 base policy 在 $t_0$ 前向，生成 nominal chunk $A_{t_0}^{base}$
3. 选择 offset $k$（chunk 已执行步数），得到当前时刻 $t = t_0 + k$
4. 构造三个核心对象：
  - 当前观测 $(o_t, s_t)$
  - 剩余 nominal chunk $A_t^{base,rem} = [a_{t_0+k}^{base}, \ldots, a_{t_0+H-1}^{base}]$
  - GT future chunk $A_t^{gt} = [a_t^{gt}, \ldots, a_{t+H_{rem}-1}^{gt}]$
5. 生成伪标签：


| 伪标签                     | 计算方式                                         | 含义          |
| ----------------------- | -------------------------------------------- | ----------- |
| `delta_action_norm`     | $\Delta A_t = A_t^{gt} - A_t^{base,rem}$ 的范数 | 每个时间步的纠偏幅度  |
| `trigger_label`         | $\max_j D_j > \epsilon$                      | 是否需要纠偏（0/1） |
| `risk_score`            | $\max(\text{deltanorm})$                     | 风险强度        |
| `correction_mask`       | deltanorm ≥ Q75 阈值                           | 哪些时间步需要纠偏   |
| `affected_region_prior` | deltanorm 的归一化分布                             | 受影响区域的软分布   |


**权威构建脚本**：`tools/build_fasa_dataset.py`

### 4.3 当前工程化升级

在基础伪标签之上，当前又增加了两层结构：


| 升级                    | 文件                        | 作用                               |
| --------------------- | ------------------------- | -------------------------------- |
| `region_target_15`    | `build_fasa_dataset.py`   | 将变长 region 目标统一重采样到固定长度 15       |
| `a_outputs` (sidecar) | `build_fasa_a_outputs.py` | 构建 A 模块的 standalone 输入，固定 7 字段契约 |


### 4.4 `a_outputs` 七字段契约

这是 A 模块 standalone 模式下的输入/输出标准格式：


| 字段                  | 类型                  | 含义                  |
| ------------------- | ------------------- | ------------------- |
| `risk_pred`         | float               | 纠偏风险分数              |
| `trigger_logit`     | float               | 纠偏触发 logit（>0 表示触发） |
| `delta_pred`        | float               | 纠偏幅度预测              |
| `region_logits`     | list[float], len=15 | 受影响时间区域的 logits     |
| `dynamic_embedding` | list[float], len=16 | 动态环境表征嵌入            |
| `version`           | string              | schema 版本标签         |
| `source`            | string              | 生成器来源标识             |


### 4.5 伪标签升级：从代理标签到结果标签（Outcome Labels）

> 详细设计文档：`docs/starvla_retrofit/handoff/phase3_0b_outcome_label_upgrade_spec.md`

**当前伪标签存在的核心问题**：

当前 `risk / trigger / region` 的语义主要来自相邻动作差分 `delta_t`，而不是"未来执行结果"。这导致三类结构性缺陷：


| 问题             | 说明                                                                                    |
| -------------- | ------------------------------------------------------------------------------------- |
| **风险定义错位**     | `risk_score = max(delta_norm)` 学到的是"动作变化幅度"而不是"失败风险"。低动作差但高失败风险的样本（如观测延迟导致抓空）会被漏判     |
| **trigger 退化** | `trigger_label = max(delta_norm >= q75)`，只要 deltanorm 非空几乎总为 1；全零动作差分也会得到 trigger=1   |
| **region 被污染** | `region_target_15` 来自 `max(prior, mask)` 的重采样，一旦 correctionmask 退化为全 1，region 也失去定位能力 |


**升级目标（Phase3.0B）**：

将标签语义从"动作差分代理"升级为"未来结果定义"：

- `risk_score` → 未来窗口 $H$ 内的失败/失稳风险
- `trigger_label` → 是否需要介入/重规划
- `delta_action_norm` → 最小必要纠偏强度（而非动作内部抖动）
- `region_target_15` → 未来 chunk 中真正与不良结果相关的时间区域

**推荐方案：Hybrid Outcome Labels**

定义未来窗口 $H$（建议起始 $H=4$），对每个未来步 $k \in [1, H]$：

$$
f_k = \mathbb{1}[\text{到 } t+k \text{ 时已失败}], \quad d_k = \text{clip}\left(\frac{s_{exec}(t+k) - s_{ref}(t+k)_W}{\tau_s},\ 0,\ 1\right)
$$

$$
\text{riskscore} = \text{clip}\left(\max\left(\max_k f_k,\ q_{90}(d_{1:H})\right),\ 0,\ 1\right)
$$

$$
\text{triggerlabel} = \mathbb{1}\left[\max_k f_k = 1\ \text{or}\ q_{90}(d_{1:H}) > \alpha_d\right], \quad \alpha_d = 0.35
$$

$$
\text{regiontarget15}[i] = \max_{k \in \text{bin}_i} c_k, \quad c_k = \max(f_k, d_k)
$$

**关键设计原则**：

1. 保持字段名兼容，减少主链路改动
2. 先改构建逻辑和审计逻辑，不先改模型结构
3. 先做监督学习，不先上 RL
4. `future_state` 已在数据中回填，应真正参与标签定义

**升级验收标准**：


| 指标                            | 门限       |
| ----------------------------- | -------- |
| 零差分样本误触发率                     | < 1%     |
| `trigger_label` 正样率           | 1% ~ 30% |
| `region_target_15` 平均激活 bin 数 | < 5/15   |
| 500-step 所有分解 loss 有限且非退化     | 全通过      |
| `Success@LIBERO` 不劣于当前基线      | > -2pp   |


**当前状态**：Phase3.0B 伪标签设计已完成 v0.9.1（含自适应超参数标定），LIBERO 远程验证 AH-1~AH-3 全部通过。训练损失改进（v0.9.2）已设计完毕，待 A-BUILD 实施。

### 4.6 为什么我们的伪标签学习像 VAE？——后验推断视角

> 本节帮助你理解"伪标签监督学习"背后的统计学本质。不需要数学背景，用直觉就能看懂。

#### 什么是 VAE（变分自编码器）

想象你有一堆手写数字图片。VAE 做两件事：

1. **编码器**：看一张图片 → 猜出它背后的"风格"（比如"偏瘦的 3"、"偏胖的 7"）。这些风格用一组数字（潜变量 $z$）表示。
2. **解码器**：拿到风格数字 → 生成一张新图片。

训练时，VAE 不是只让编码器猜一个数（**点估计**），而是让它输出一个**分布**（均值 $\mu$ 和方差 $\sigma^2$），表达"我认为风格大概在这个范围内"。这就是**后验推断**——给定观测，推断隐藏变量的概率分布。

#### 我们的 A 模块在做什么

A 模块面对同样的问题：

- **观测**：当前相机画面、机器人状态、最近的动作历史
- **隐藏变量**：未来会不会出问题（risk）、需不需要纠偏（trigger）、哪些时间步受影响（region）
- **推断**：在推理时看不到未来，但需要猜出上述隐藏变量

用表格来对比：


| 概念   | VAE 中的含义       | A 模块中的含义                          |
| ---- | -------------- | --------------------------------- |
| 观测   | 手写数字图片         | 当前观测 $(o_t, s_t)$、动作历史            |
| 隐藏变量 | 图片的"风格"        | 未来是否需要纠偏、风险有多大、哪里出问题              |
| 编码器  | 看图 → 猜风格分布     | A 模块 → 预测 risk / trigger / region |
| 训练信号 | 让解码器从猜出的风格重建图片 | 伪标签（通过 future_state 提前计算好的答案）     |


#### 关键区别：为什么当前系统不完美


| 维度    | VAE                                     | 当前 A 模块                       |
| ----- | --------------------------------------- | ----------------------------- |
| 输出什么  | 分布（均值+方差）——"我觉得大概是 0.6，但可能在 0.3~0.9 之间" | 点估计（单个数）——"risk=0.6"，没有置信度    |
| 各输出之间 | 通过 KL 散度正则化，保证协调                        | 各头独立——trigger 和 region 可能"打架" |
| 反馈循环  | 解码器重建误差会反馈给编码器                          | 伪标签离线固定，不随模型进步更新              |


这就是为什么我们需要后续改进。

#### 我们的改进计划

基于上述分析，改进分阶段推进：

**当前轮次（v0.9.2，已设计）**：

1. **Region Loss 条件化**（P0）：只在 trigger=1 的样本上计算 region 损失。理由：trigger=0 意味着"不需要纠偏"，此时 region 的正确答案是"没有受影响区域"。在这些样本上训练 region 只会引入噪声。
2. **一致性正则**（P1）：添加一个约束，要求 trigger 和 region 不能矛盾。如果 region 认为很多时间步都受影响，那 trigger 也应该认为需要纠偏。这借鉴了 VAE 中 KL 正则的思想——强制各输出头之间保持一致。

**下一轮次（规划中）**：

1. **引入置信度**：让 risk 头输出不只是一个数，而是"均值+方差"（像 VAE 一样输出分布）。这样下游 corrective policy 可以根据 A 模块的"自信程度"来决定纠偏力度。
2. **解决数据分布偏差**：当前伪标签基于专家示教数据构建，但 base policy 执行时的状态分布与专家不同。长期解决方案是引入强化学习，让系统在真实执行中持续改进。

> 详细研究分析见：`docs/algorithm1/handoff/research_notes/phase3_0b_vae_accerl_analysis.md`

### 4.7 关于 AcceRL 框架的启发

AcceRL 是一个将**世界模型**与**异步强化学习**结合来改进 VLA 模型的框架。简单来说：

- 它训练一个"世界模型"来预测"如果我做了动作 A，世界会变成什么样"
- 然后在世界模型中"想象"大量轨迹来训练策略（不需要真实机器人交互）
- 同时在真实环境中异步采集数据，不断校正世界模型

**对我们的启发**：


| AcceRL 做法  | 对应到我们的系统                                           |
| ---------- | -------------------------------------------------- |
| 世界模型生成虚拟数据 | 我们可以通过状态扰动模拟偏差（数据增强）                               |
| 密集奖励信号     | 我们的 risk_score 可以作为未来 corrective policy RL 训练的密集奖励 |
| 在线交互消除分布偏差 | 长期目标：DAgger 式迭代训练                                  |


**一个重要结论**：我们此前尝试的"动作随机加噪"效果不佳，因为噪声加在动作上但伪标签基于状态计算——两者不一致。未来如果做数据增强，应该是在**状态**上加扰动，或者直接用强化学习获取真实偏差数据。

---

## 五、A 模块详解

### 5.1 定位

A 模块（Dynamic Change Prior Module）负责回答三个问题：

1. **是否需要纠偏？** → `trigger_logit`
2. **风险多大？** → `risk_pred`
3. **哪些时间步受影响？** → `region_logits`
4. **环境动态表征是什么？** → `dynamic_embedding`

### 5.2 两种运行模式


| 模式           | 实现类                          | 数据来源                     | 适用场景            |
| ------------ | ---------------------------- | ------------------------ | --------------- |
| `lite`       | `LiteAModuleInterface`       | 模型内轻量头预测                 | 联合训练 / fallback |
| `standalone` | `StandaloneAModuleInterface` | 读取外部 `a_outputs` payload | 独立 A 模块训练       |


代码位置：`starVLA/model/framework/a_module_interface.py`

### 5.3 训练范式

**输入**：

$$
X_t^A = (o_{t-K:t},\ s_{t-K:t},\ a_{t-K:t-1})
$$

即最近 $K$ 帧的观测窗口、状态窗口、动作历史。

**监督目标**（来自伪标签）：


| 监督信号                | 来自                                                        | 损失函数            |
| ------------------- | --------------------------------------------------------- | --------------- |
| `risk_score`        | `pseudo_labels.risk_score`                                | MSE             |
| `trigger_label`     | `pseudo_labels.trigger_label`                             | BCE with logits |
| `delta_action_norm` | `pseudo_labels.delta_action_norm`                         | MSE             |
| `region_target`     | `pseudo_labels.affected_region_prior` + `correction_mask` | BCE + MSE       |
| `dynamic_embedding` | `a_outputs.dynamic_embedding`                             | MSE             |


**分解损失**（在 `QwenPI.py` 中实现）：


| 损失名                      | 含义      |
| ------------------------ | ------- |
| `a_loss_risk`            | 风险分数预测  |
| `a_loss_trigger`         | 纠偏触发预测  |
| `a_loss_embed`           | 动态嵌入预测  |
| `corrective_loss_delta`  | 纠偏幅度预测  |
| `corrective_loss_region` | 受影响区域预测 |


### 5.4 训练命令示例

```bash
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_pi.yaml \
  --datasets.vla_data.shared_builder_enabled true \
  --datasets.vla_data.correction_supervision_enabled true \
  --datasets.vla_data.correction_dataset_jsonl <数据集路径> \
  --trainer.optional_loss_hooks.enabled true \
  --trainer.optional_loss_hooks.a_loss.enabled true \
  --trainer.optional_loss_hooks.corrective_loss.enabled true \
  --framework.a_module.mode standalone \
  --framework.a_module.standalone.input_key a_outputs \
  --framework.a_module.standalone.output_contract.strict_shape true \
  --framework.a_module.standalone.output_contract.expected_region_len 15 \
  --framework.a_module.standalone.output_contract.expected_embedding_dim 16 \
  --trainer.max_train_steps 5000 \
  --run_id a_module_train_$(date +%Y%m%d_%H%M%S)
```

### 5.5 训练前必做：数据审计

```bash
python tools/a_outputs_audit.py \
  --input-jsonl <数据集路径> \
  --expected-region-len 15 \
  --expected-embedding-len 16 \
  --require-a-outputs \
  --strict
```

---

## 六、Corrective Policy 详解

### 6.1 定位

Corrective Policy（与 residual policy 等价）负责：**在 A 模块判断需要纠偏时，输出具体的纠偏动作 delta**。

最终目标：

$$
A_t^{corrected} = A_t^{base,rem} + \Delta A_t
$$

### 6.2 输入

$$
X_t^{corr} = (o_t,\ s_t,\ A_t^{base,rem},\ z_t^{dyn},\ r_t,\ m_t^{prior},\ \tau_t)
$$


| 符号               | 含义               | 来源          |
| ---------------- | ---------------- | ----------- |
| $o_t, s_t$       | 当前观测和状态          | 实时传感器       |
| $A_t^{base,rem}$ | 剩余 nominal chunk | base policy |
| $z_t^{dyn}$      | 动态嵌入             | A 模块输出      |
| $r_t$            | 风险分数             | A 模块输出      |
| $m_t^{prior}$    | 受影响区域先验          | A 模块输出      |
| $\tau_t$         | 阶段/时间索引          | 可选          |


### 6.3 输出


| 输出                         | 含义          |
| -------------------------- | ----------- |
| `trigger` $g_t$            | 是否执行纠偏（0/1） |
| `delta_chunk` $\Delta A_t$ | 纠偏动作增量      |
| `refined_mask`（可选）         | 精细化的受影响区域   |


### 6.4 训练阶段划分


| 阶段                  | 策略                          | 说明                          |
| ------------------- | --------------------------- | --------------------------- |
| **3B-1 Warm-up**    | 用 oracle risk / oracle mask | 不依赖 A 模块，先验证 policy 本身的学习能力 |
| **3B-2 Transition** | 混合 oracle 和 A 输出            | 逐步从 GT 切到预测                 |
| **3B-3 Final**      | 仅用 A 真实输出                   | 端到端闭环验证                     |


### 6.5 损失函数

$$
\mathcal{L}*{corr} = \lambda_1 \mathcal{L}*{\Delta} + \lambda_2 \mathcal{L}*{trig} + \lambda_3 \mathcal{L}*{smooth}
$$

- $\mathcal{L}_{\Delta}$：delta chunk 重建误差
- $\mathcal{L}_{trig}$：trigger 分类损失
- $\mathcal{L}_{smooth}$：纠偏动作平滑性正则

### 6.6 关键约束

**corrective policy 的训练和启动必须等 A 模块输出稳定后才能开始**。在本仓库的 Harness v2 体系中，这通过 `A-REVIEW → CP-BUILD` 的 upstream gate 强制执行。

---

## 七、分阶段训练总流程

```text
Phase 0                Phase 1                 Phase 2              Phase 3
─────────────         ─────────────           ─────────────        ─────────────
训练 Base Policy       构建 Correction          训练 A 模块           训练 Corrective
                       Dataset                                      Policy
     │                      │                       │                    │
     ▼                      ▼                       ▼                    ▼
base chunk policy     pseudo_labels            risk_pred             trigger
能输出 H 步            a_outputs                trigger_logit         delta_action
action chunk          region_target_15         region_logits
                                               dynamic_embedding
                                                    │
                                               冻结 A 模块
                                                    │
                                               A-REVIEW 验收
                                                    │
                                               放行给 Phase 3
                                                    ▼
                                               CP 消费 A 输出
```

**关键顺序依赖**：

- Phase 1 依赖 Phase 0 的 base policy
- Phase 2 依赖 Phase 1 的 correction dataset
- Phase 3 依赖 Phase 2 的 A 模块输出 + A-REVIEW 验收放行
- 可选 Phase 4：A + Corrective Policy 联合微调（backbone 冻结）

---

## 八、代码结构导航

### 8.1 核心代码文件


| 文件                                               | 职责                                 |
| ------------------------------------------------ | ---------------------------------- |
| `starVLA/model/framework/QwenPI.py`              | 主模型前向、A-loss 分解、corrective-loss 分解 |
| `starVLA/model/framework/a_module_interface.py`  | A 模块接口（lite / standalone 模式切换）     |
| `starVLA/model/framework/optional_loss_utils.py` | 伪标签 → 监督张量的解析与构造                   |
| `starVLA/training/train_starvla.py`              | 训练入口脚本                             |


### 8.2 数据构建工具


| 文件                              | 职责                                         |
| ------------------------------- | ------------------------------------------ |
| `tools/build_fasa_dataset.py`   | 统一伪标签构建器（pseudo_labels + region_target_15） |
| `tools/build_fasa_a_outputs.py` | A 模块 sidecar 构建器（a_outputs 七字段）            |
| `tools/build_vdpm_a_outputs.py` | 外部 VDPM 结果桥接适配器                            |


### 8.3 数据审计工具


| 文件                                    | 职责                      |
| ------------------------------------- | ----------------------- |
| `tools/a_outputs_audit.py`            | a_outputs 字段完整性/形状审计    |
| `tools/fasa_a_outputs_audit.py`       | FASA 数据集 a_outputs 专项审计 |
| `tools/fasa_dataset_audit.py`         | FASA 数据集整体审计            |
| `tools/train_fasa_a_outputs_check.py` | 训练前 a_outputs 预检        |
| `tools/train_fasa_sidecar_smoke.py`   | sidecar 数据冒烟测试          |


### 8.4 Shared Builder Schema

数据契约版本：`p1_shared_builder_v1`


| 关键字段                    | 说明                         |
| ----------------------- | -------------------------- |
| `action_chunk_shape`    | `[T, D]`，通常 T=8, D=7       |
| `meta.schema_version`   | 必须为 `p1_shared_builder_v1` |
| `meta.action_chunk_len` | 必须等于 T                     |
| `meta.action_dim`       | 必须等于 D                     |


验证工具：`tools/handoff/validate_shared_builder_schema.py`

---

## 九、训练输出与验收

### 9.1 训练输出目录

```text
<run_root_dir>/<run_id>/
├── config.yaml        # 训练配置快照
├── metrics.jsonl      # 逐步指标日志
├── summary.jsonl      # 汇总指标
├── train.log          # 训练日志
└── checkpoints/       # 模型检查点
```

### 9.2 关键监控指标

**A 模块侧**：


| 指标             | 健康标准      |
| -------------- | --------- |
| `loss/risk`    | 有限值，逐步下降  |
| `loss/trigger` | 有限值，逐步下降  |
| `loss/delta`   | 非全零，有限值   |
| `loss/region`  | 非全零，有限值   |
| `loss/embed`   | 非全零，有限值   |
| `loss/total`   | 有限值，无 NaN |


**Corrective Policy 侧**：


| 指标                           | 健康标准    |
| ---------------------------- | ------- |
| delta chunk 重建误差             | 逐步下降    |
| trigger 分类准确率                | 逐步提升    |
| correction smoothness        | 保持在合理范围 |
| 相比无 correction 的 improvement | 正向      |


### 9.3 最终验收硬门控

- 主验收指标：`Success@LIBERO`
- 辅助约束：`p95 latency` 和稳定性（无 nonfinite / crash）
- 诊断指标（loss / mask / delta）是支撑证据，不是最终 pass/fail 门控

---

## 十、并行开发体系（Harness v2）

当 A 模块和 corrective policy 同时推进时，使用六线程并行开发体系：


| 线程          | 模块                | 角色  | 职责               |
| ----------- | ----------------- | --- | ---------------- |
| `A-RES`     | A-module          | 研究  | 定义语义、标签、实验假设     |
| `A-BUILD`   | A-module          | 实现  | 在 manifest 范围内实现 |
| `A-REVIEW`  | A-module          | 审核  | 独立验证 + 发布下游可用性结论 |
| `CP-RES`    | corrective policy | 研究  | 定义目标函数、消费规则      |
| `CP-BUILD`  | corrective policy | 实现  | 实现训练和推理链路        |
| `CP-REVIEW` | corrective policy | 审核  | 验证是否正确消费 A 输出    |


**关键规则**：`CP-BUILD` 必须等 `A-REVIEW` 明确放行后才能启动。

详细操作流程见：`docs/starvla_retrofit/handoff/thread_operator_quick_reference.md`

---

## 十一、新人最短上手路径

### 如果你想理解项目

1. **读本文件**（你正在读的这篇）— 全景理解
2. 读 `a_module_newcomer_guide.md` — A 模块伪标签细节
3. 读 `a_module_design.md` — A 模块技术契约
4. 读 `star_vla改造与统一伪标签生成任务书.md` — 工程总任务书

### 如果你想跑训练

1. 审计数据：`python tools/a_outputs_audit.py --input-jsonl <数据> --strict`
2. 参考 `a_module_training_guide.md` 的启动模板
3. 监控 `metrics.jsonl` 中的分解损失

### 如果你想参与开发

1. 读 `thread_operator_quick_reference.md` — 了解六线程流程
2. 确定你的线程角色（RES / BUILD / REVIEW）
3. 设置 `STARVLA_THREAD_ID` 并运行 readiness 检查
4. 在 manifest 范围内工作

---

## 十二、常见问题（FAQ）

### Q1：A 模块和 corrective policy 的关系是什么？

A 模块是"诊断器"，corrective policy 是"治疗器"。A 告诉你"哪里出问题了、有多严重"，corrective policy 根据这些信息输出"怎么修正"。

### Q2：现在是不是已经实现了在线 VDPM 推理？

不是。当前 A 模块的主链路是读取离线构建的 `a_outputs`。生成器可以是启发式规则或外部预计算。在线 VDPM 推理是后续目标。

### Q3：corrective policy 可以在 A 模块训练完成前就开始开发吗？

研究（CP-RES）可以提前启动，但必须标记所有依赖 A 输出的结论为 `BLOCKED_WAIT_UPSTREAM`。实现（CP-BUILD）必须等 A-REVIEW 放行。

### Q4：`pseudo_labels` 和 `a_outputs` 有什么区别？

`pseudo_labels` 是训练监督标签（来自伪标签构建器），`a_outputs` 是 A 模块的 standalone 输入/输出载体（7 字段契约）。前者用于 loss 计算，后者用于模块间解耦。

### Q5：训练时需要同时开 A-loss 和 corrective-loss 吗？

取决于训练阶段。Phase 2 只训 A，开 `a_loss`；Phase 3 训 corrective policy，开 `corrective_loss`。具体由 config 中的 `optional_loss_hooks` 控制。

### Q6：数据格式是固定的吗？

当前契约固定为 `p1_shared_builder_v1`，`region_logits` 长度=15，`dynamic_embedding` 维度=16。如需变更，必须走版本化 contract rollout 流程。

---

## 十三、关联文档索引


| 文档                                                                        | 定位                                          |
| ------------------------------------------------------------------------- | ------------------------------------------- |
| `docs/starvla_retrofit/handoff/star_vla改造与统一伪标签生成任务书.md`                  | 工程总任务书                                      |
| `docs/starvla_retrofit/handoff/phase3_0b_outcome_label_upgrade_spec.md`   | 伪标签升级设计（outcome labels，Phase3.0B，当前 v0.9.2） |
| `docs/algorithm1/handoff/research_notes/phase3_0b_vae_accerl_analysis.md` | A-RES 研究笔记：VAE 后验推断视角与 AcceRL 对比分析          |
| `docs/algorithm1/handoff/a_module_newcomer_guide.md`                      | A 模块新手指南                                    |
| `docs/algorithm1/handoff/a_module_design.md`                              | A 模块技术设计（Phase2.2 baseline）                 |
| `docs/algorithm1/handoff/a_module_training_guide.md`                      | A 模块训练操作手册（Phase2.3）                        |
| `docs/starvla_retrofit/handoff/shared_builder_schema_contract.md`         | 共享 builder 数据契约                             |
| `docs/starvla_retrofit/handoff/thread_operator_quick_reference.md`        | 六线程操作者快速参考                                  |
| `docs/starvla_retrofit/handoff/harness_v2_overview.md`                    | 并行开发体系架构                                    |
| `docs/starvla_retrofit/handoff/harness_v2_thread_matrix.md`               | 六线程矩阵定义                                     |
| `docs/algorithm1/handoff/progress_live.md`                                | 全局进度主日志                                     |


