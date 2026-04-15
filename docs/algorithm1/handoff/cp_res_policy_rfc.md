# Corrective Policy RFC — delta_action_v0_draft

```
status:     DRAFT
author:     CP-RES thread
thread_id:  CP-RES
round_id:   CP-ROUND-BOOTSTRAP-WAIT-UPSTREAM
gate:       phase3_0b_cp_research
date:       2026-04-15
handoff_to: CP-BUILD, CP-REVIEW
```

---

## 0. 语义等价声明

本仓中 **corrective policy** 与 **residual policy** 语义完全等价。后文统一使用 corrective policy。

---

## 1. 目标函数定义

### 1.1 问题描述

base policy（action chunk）在执行过程中会产生状态偏差 $d_k$（参见 spec §3.2）。  
corrective policy 的目标是：**给定当前观测和 A-module 的评估输出，预测何时、何处、以多大幅度对 base action 施加修正**。

v0 draft 的实现范围限定为 **修正元数据预测**（correction metadata prediction），而非直接回归动作偏移向量。这与现有 `QwenPI.py` 架构的 `corrective_delta_head` / `corrective_region_head` 一致。

### 1.2 形式化目标

设 $o_t$ 为当前观测，$A_t$ 为 A-module 在时刻 $t$ 的输出：

$$
A_t = \{\text{risk\_pred},\ \text{trigger\_logit},\ \text{delta\_pred},\ \text{region\_logits}\}
$$

corrective policy 学习以下四个头：

| 头 | 预测量 | 训练目标来源 | 损失类型 |
|---|---|---|---|
| `risk_head` | $\hat{r} \in [0,1]$ | `pseudo_labels.risk_score` | MSE |
| `trigger_head` | $\hat{t} \in \mathbb{R}$ (logit) | `pseudo_labels.trigger_label` | BCE |
| `delta_head` | $\hat{\delta} \in [0,1]$ | `pseudo_labels.delta_action_norm` | MSE（mask: trigger=1）|
| `region_head` | $\hat{\rho} \in \mathbb{R}^{C}$（C = chunk\_len） | `pseudo_labels.correction_mask` + `.affected_region_prior` | BCE + MSE（P0 conditional: trigger=1 only）|

其中 $C$ = `chunk_len`（训练时默认 8，region 目标由 `_coerce_vector` 从 `horizon_steps=4` 零填充到 8）。

### 1.3 训练损失结构

沿用 A-module 的 `corrective_loss` 结构（已在 `QwenPI.py` 中实现），并加入 v0.9.2 改进：

```
corrective_loss = w_risk    × risk_loss            (all samples)
               + w_trigger  × trigger_loss         (all samples)
               + w_delta    × delta_loss           (trigger=1 samples only)
               + w_region   × region_loss_P0       (trigger=1 samples only, per P0)
               + w_consist  × consist_loss         (all samples, per P1)
```

**P0（已实现）**：`region_loss` 仅在 `trigger_label > 0.5` 的样本上计算，消除 trigger=0 样本的虚假 region 监督。

**P1（已实现）**：`consist_loss` 惩罚 "region 高激活 + trigger 低概率" 的语义矛盾，单向 hinge 设计。

### 1.4 推理时修正机制（v0 草案）

推理时使用 `LiteAModuleInterface`（当前默认）或未来的 `StandaloneAModuleInterface`：

1. 前向传播获得 A-module 预测：`trigger_prob = sigmoid(trigger_logit)`
2. 若 `trigger_prob > θ_trigger`（默认 0.5）：激活修正
3. 修正幅度参考 `delta_pred`，修正区域参考 `sigmoid(region_logits)`
4. **v0 的实际 action 混合机制由 CP-BUILD 设计**，本 RFC 不预设。

> CP-BUILD 决策：是否需要独立的 "corrective action chunk" 输出头？  
> 若需要，CP-RES 在 Round 2 补充相应设计；v0 先以元数据预测为主。

---

## 2. 训练范式

### 2.1 数据来源

| 数据集 | 状态 | 位置 |
|---|---|---|
| FASA v0.9.3 LIBERO 2000 samples | BUILT（远程） | `results/PseudoLabels/p3_0b_outcome_v093/` |
| 训练配置 | 待 CP-BUILD 产出 | — |

### 2.2 训练模式

**Phase 3（P3）corrective trainer**：

- 训练对象：`corrective_delta_head` + `corrective_region_head`（及配套 risk/trigger heads）
- 是否冻结其余参数：由 CP-BUILD 决定（建议先联合训练，后期可考虑冻结 backbone）
- 数据 pipeline：复用 P1/P2 已验证的 FASA dataloader
- 监控指标：`corrective_loss_region`、`corrective_loss_delta`、`a_loss_consist`

### 2.3 验收假设（CP-AH）

| # | 指标 | 门限 | 测量方式 |
|---|---|---|---|
| CP-AH-1 | trigger=0 样本的 region loss = 0（P0 生效验证） | 100% | 检查 `debug/corrective_loss_region_trigger_filtered_count` |
| CP-AH-2 | 500-step 训练所有 corrective loss 分量有限且非退化 | 全通过 | `metrics.jsonl` 中 corrective_loss_* 全为 finite |
| CP-AH-3 | `a_loss_consist` 有限且逐步下降 | 趋势下降 | `metrics.jsonl` |
| CP-AH-4 | `Success@LIBERO` 不劣于 A-only 基线 | > -2pp | 远程评估（BLOCKED_WAIT_REMOTE） |

CP-AH-1 ~ CP-AH-3 可在 CP-BUILD 本地 smoke test 阶段验证；  
CP-AH-4 需要在远程服务器执行 LIBERO eval。

---

## 3. delta_action_v0_draft 输出契约

CP-RES 产出的 `delta_action_v0_draft` 契约定义如下：

```yaml
contract_id: delta_action_v0_draft
version: v0
status: DRAFT
date: 2026-04-15

outputs:
  - field: trigger_prob
    type: float
    range: [0.0, 1.0]
    semantics: P(correction needed) = sigmoid(trigger_logit)
    
  - field: delta_pred
    type: float
    range: [0.0, 1.0]
    semantics: predicted correction magnitude (normalized, outcome-mixed)
    
  - field: region_logits
    type: list[float]
    length: chunk_len (default 8)
    semantics: per-step region correction logits, sigmoid gives correction probability per step
    
  - field: risk_pred
    type: float
    range: [0.0, 1.0]
    semantics: predicted risk level of current state

training_targets_source: pseudo_labels (FASA v0.9.3)

inference_mode: lite (current default)
inference_mode_note: >
  In "lite" mode, all four heads share the VLM backbone's pooled_hidden.
  In future "standalone" mode, A-module predictions from a pre-trained checkpoint
  would be read from sample payload via StandaloneAModuleInterface.

action_correction_mechanism: TBD (CP-BUILD v0 decision)
```

---

## 4. 冻结锚点确认

| 锚点 | 值 | 来源 |
|---|---|---|
| VLM | qwen25_authoritative | 全局锚点 |
| A-output contract | v0.9.3_frozen | A-REVIEW PASS_CONFIRMED (2026-04-15T00:30:00+08:00) |
| schema | p1_shared_builder_v1 | P1 builder |
| 训练消费接口 | `optional_loss_utils.py::build_optional_hook_targets()` | 全文件冻结，不可修改 |
| A-module interface | `a_module_interface.py` | 全文件冻结，不可修改 |

---

## 5. 开放风险

| # | 风险 | 严重度 | 备注 |
|---|---|---|---|
| CP-R1 | v0 不输出实际 Δa 向量，推理时修正依赖 CP-BUILD 设计的混合机制，机制未定 | MEDIUM | 由 CP-BUILD 在 v0 round 中决策 |
| CP-R2 | `correction_mask` 从 horizon_steps=4 零填充到 chunk_len=8，有效监督率 50% | LOW | 继承自 A-REVIEW OBS-2；已知可接受 |
| CP-R3 | FASA v0.9.3 仅 2000 samples LIBERO，泛化到其他 task 未验证 | LOW | AH-4 远程验证后评估 |
| CP-R4 | `region_target_15`（15-bin）未被当前训练代码消费，CP 侧若需使用需明确声明 | INFO | 当前不消费；若 CP-BUILD 需要，在 Round 2 补充 |

---

## 6. 交付总结

| 工件 | 状态 | 路径 |
|---|---|---|
| Policy RFC | DELIVERED | `docs/algorithm1/handoff/cp_res_policy_rfc.md` |
| Interface Assumptions | DELIVERED | `docs/algorithm1/handoff/cp_res_interface_assumptions.md` |
| Blocked Items | DELIVERED | `docs/algorithm1/handoff/cp_res_blocked_items.md` |

**交付对象**：CP-BUILD、CP-REVIEW
