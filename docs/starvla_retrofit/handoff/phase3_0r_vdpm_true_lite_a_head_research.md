# Phase3.0R: VDPM 引入下“真正 lite（非 MLP-only）A-head”架构调研与方案收敛

Status: R-thread research handoff  
Date: 2026-04-12 (Asia/Shanghai)  
Scope: `VDPM frozen embedding + Qwen pooled_hidden` 图内可训练 A-head 融合网络  
Codebase anchor: `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410`

---

## 1) 结论先行

推荐唯一方案：

- 采用 `dynamic_embedding-only` 的 `Q-Bridge Cross-Attention A-head` 作为主线：
  - `VDPM runtime_model` 保持冻结，继续以 in-loop `runtime_model` 方式运行；
  - 主模型图内新增 query-based 融合头，使用 `Qwen pooled_hidden + VDPM dynamic_embedding(16)` 做融合；
  - 输出继续严格遵守 frozen contract：
    - `risk_pred`
    - `trigger_logit`
    - `delta_pred`
    - `region_logits(len=15)`
    - `dynamic_embedding(len=16)`

保底备选方案：

- `Tiny Transformer Encoder Fusion A-head`

不推荐主线继续沿用当前 `pooled_hidden -> 多个独立 MLP` 的 `lite` 头；该路径不能满足“真正融合网络”的目标。

---

## 2) 约束基线

本方案必须同时对齐以下已冻结事实：

### 2.1 Frozen contract

见：

- `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/p4_1_qwen25_longrun_authoritative_baseline.md`

冻结字段为：

- `risk_pred`
- `trigger_logit`
- `delta_pred`
- `region_logits(len=15)`
- `dynamic_embedding(len=16)`
- `version`
- `source`

### 2.2 Frozen mainline / non-retreat rules

同文档明确要求：

- 不放松 strict contract
- 不启用 `fallback_to_lite`
- 不把新路径伪装成“成功”
- 不把 Qwen3 作为下游主线替代

### 2.3 当前训练与日志语义

现有 trainer 已冻结以下 loss/logging 键语义，不允许破坏：

- `loss/risk`
- `loss/trigger`
- `loss/delta`
- `loss/region`
- `loss/embed`
- `loss/a_module`
- `loss/corrective`
- `loss/total`

对应实现：

- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/training/train_starvla.py`
- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/model/framework/QwenPI.py`
- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/model/framework/optional_loss_utils.py`

### 2.4 现有真实 in-loop 证据基线

Phase2.5b 029 已证明：

- `vdpm_mode=inloop`
- `vdpm_source=runtime_model`
- `vdpm_model_calls_total=21 > 0`
- `fallback_to_precomputed_count=0`
- `train_50/500` 全 finite
- `delta/embed` 非零门禁通过
- `eval smoke` 通过

归档工件：

- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/_remote_runs/p4_1_fasa_phase2_5b_real_inloop_20260412/20260412_102443__ALG1-FASA-20260412-029-OC`

---

## 3) 当前代码现状与关键观察

### 3.1 当前 `QwenPI` 中的 A-lite 仍是 MLP-only

当前 `QwenPI.__init__` 中：

- `a_risk_head`
- `a_trigger_head`
- `corrective_delta_head`
- `corrective_region_head`
- `a_embed_head`

全部是：

```text
LayerNorm -> Linear -> SiLU -> Linear
```

即典型 `pooled_hidden -> head` 结构，不是融合网络。

关键位置：

- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/model/framework/QwenPI.py`

### 3.2 当前 standalone / in-loop runtime 路径已经可复用

`a_module_interface.py` 已具备以下能力：

- `lite`
- `standalone`
- `inloop + runtime_infer`
- `inloop + runtime_model`
- runtime worker
- cache
- 审计指标：
  - `inloop_calls`
  - `inloop_success`
  - `inloop_fail`
  - `fallback_to_precomputed_count`
  - `vdpm_model_calls_total`
  - `cache_hit_ratio`

因此 Phase3.0R 的工程重点不该再放在“如何把 VDPM 接进来”，而应放在“如何把 frozen VDPM 特征喂给图内 trainable fusion head”。

### 3.3 当前最危险的张力：contract `region=15` vs train-time `chunk_len=8`

当前 config：

- `future_action_window_size=7`
- `past_action_window_size=0`
- `chunk_len=8`

而 frozen contract 要求：

- `region_logits(len=15)`

现有 standalone 读取路径会把 region 目标按 `chunk_len` 训练消费，因此如果直接把 contract `15` 和 train-time `8` 混用，很容易出现隐式截断/对齐不清。

结论：

- Phase3.0R 必须显式引入 `region_contract15 -> region_train(chunk_len)` adapter。
- 不能继续依赖隐式 `coerce_vector(..., target_len=self.chunk_len)` 的行为。

---

## 4) 候选架构对比

统一估算基准：

- `Hq = 2048`
- `Df = 16`
- `D = 256`
- `chunk_len = 8`
- contract `region = 15`

所有候选都满足：

- 输入必须包含：
  - `Qwen pooled_hidden`
  - `VDPM frozen embedding`
- 输出必须包含：
  - `risk_pred[B]`
  - `trigger_logit[B]`
  - `delta_pred[B]`
  - `region_logits_contract15[B,15]`
  - `dynamic_embedding_pred[B,16]`

### 4.1 Candidate A: Q-Bridge Cross-Attention A-head

结构图：

```text
pooled_hidden[2048] -> q_proj -> q_token[256]
vdpm_embed[16] -> split 4x4 -> slot_proj -> 4 vdpm slots[256]
memory = [q_token, vdpm_global, 4 vdpm slots]  # 6 tokens

queries:
  - 1 global query
  - 15 region queries
  - 4 dynamic queries

2-layer cross-attn decoder
  query <- memory

heads:
  global -> risk_pred / trigger_logit / delta_pred
  region queries -> region_logits_contract15
  dynamic queries -> dynamic_embedding_pred16

adapter:
  region_logits_contract15 -> region_logits_train[chunk_len]
```

输入/输出 shape：

- `pooled_hidden`: `[B, 2048]`
- `vdpm_dynamic_embedding`: `[B, 16]`
- `memory`: `[B, 6, 256]`
- `queries`: `[B, 20, 256]`
- `risk_pred`: `[B]`
- `trigger_logit`: `[B]`
- `delta_pred`: `[B]`
- `region_logits_contract15`: `[B, 15]`
- `region_logits_train`: `[B, 8]`
- `dynamic_embedding_pred`: `[B, 16]`

参数量级：

- 约 `1.5M ~ 1.7M`

推理延迟估算：

- 额外 MACs 约 `17M`
- 在 GPU 上通常 `< 1ms`
- 显著低于 029 中 runtime miss 的 `~1.49s` 量级

梯度路径：

- 训练：
  - `q_proj`
  - `slot_proj`
  - cross-attn / FFN
  - global head
  - region head
  - dynamic head
  - region adapter
- 冻结：
  - VDPM runtime model
- 条件可训练：
  - Qwen backbone 取决于 `freeze_modules`

与当前代码对接点：

- `QwenPI.__init__`
- `QwenPI._compute_optional_hook_outputs`
- `a_module_interface.build_a_module_interface`
- `a_module_interface.StandaloneAModuleInterface`

结论：

- 推荐主线

### 4.2 Candidate B: Tiny Transformer Encoder Fusion A-head

结构图：

```text
[cls, q_token, vdpm_global, 4 vdpm slots, 1 global query, 15 region queries, 4 dynamic queries]
  -> 2-layer tiny transformer encoder
  -> read designated tokens
```

输入/输出 shape：

- 序列长度约 `26`
- 其余输出同 Candidate A

参数量级：

- 约 `1.5M ~ 1.7M`

推理延迟估算：

- 额外 MACs 约 `22M`

梯度路径：

- 与 Candidate A 相同

与当前代码对接点：

- 与 Candidate A 相同

结论：

- 可作为保底备选
- 工程实现略简单，但输出语义不如 query decoder 直接

### 4.3 Candidate C: Gated Fusion / FiLM A-head

结构图：

```text
q_proj(pooled_hidden) = q
z_proj(vdpm_embed) = z
gate = sigmoid(W[q;z])
fused = gate * q + (1-gate) * z_up

global head + region head + embed head
```

输入/输出 shape：

- `q`: `[B, 256]`
- `z`: `[B, 256]`
- `fused`: `[B, 256]`

参数量级：

- 约 `0.8M ~ 0.9M`

推理延迟估算：

- `2M ~ 3M` MACs

梯度路径：

- 同上

与当前代码对接点：

- 同上

结论：

- 最便宜
- 但结构先验最弱，容易退回“加强版 MLP”
- 不建议作为唯一主线

### 4.4 Candidate D: Perceiver Bottleneck A-head

结构图：

```text
memory = [q_token, vdpm slots]
8 latent tokens
latent <- memory cross-attn
latent self-attn
query decode -> outputs
```

参数量级：

- 约 `2.0M ~ 2.2M`

推理延迟估算：

- `12M ~ 14M` MACs

结论：

- 可扩展
- 对当前 frozen `16 + pooled_hidden` 的信息规模来说偏重

---

## 5) 最终推荐方案详解

### 5.1 为什么推荐 Q-Bridge Cross-Attention

推荐理由：

1. 它是“真正融合网络”而不是“大一点的 MLP”
2. 它与当前偏好的 query-based output semantics 对齐：
   - `1 global`
   - `15 region`
   - `4 dynamic`
3. 它只消费 `VDPM dynamic_embedding(16)`，不会学成对 frozen scalar outputs 的简单复制器
4. 工程上不需要修改 VDPM worker 协议
5. 额外计算量远小于真实 runtime_model miss 成本

### 5.2 推荐输入设计

主线只使用以下 frozen VDPM 特征：

- `dynamic_embedding(16)`

不把以下 frozen scalar 作为 fusion 主输入：

- `risk_pred`
- `trigger_logit`
- `delta_pred`
- `region_logits`

原因：

- 如果把这些也喂进图内 head，模型很容易学成“把 frozen 输出重新投影回同名输出”；
- 那样既不是真正融合，也不利于证明主模型图内 trainable A-head 的价值。

### 5.3 推荐结构

```text
VDPM runtime_model (frozen, in-loop, cached, out-of-graph)
  -> payload.dynamic_embedding[16]
  -> reshape to 4 slots x 4 dims
  -> slot_proj(4 -> 256)

Qwen pooled_hidden[2048]
  -> q_proj(2048 -> 256)

memory:
  [q_token, vdpm_global_token, vdpm_slot_1..4]

learned queries:
  - 1 global query
  - 15 region queries
  - 4 dynamic queries

2 x cross-attn blocks

outputs:
  global -> risk_pred / trigger_logit / delta_pred
  region -> region_logits_contract15
  dynamic -> dynamic_embedding_pred16

adapter:
  region_logits_contract15 -> region_logits_train[chunk_len]
```

### 5.4 region 头的 contract/train 双轨输出

推荐明确分为：

- `region_logits_contract15`
  - frozen external contract
- `region_logits_train`
  - 用于当前 `loss/region`
  - shape `[B, chunk_len]`

转换方式：

- 固定 deterministic adapter
- 推荐线性插值或固定 bin-aggregation

这样可以同时满足：

- 对外 contract `15`
- 对内当前 trainer/loss 语义不变

### 5.5 梯度路径

冻结：

- `VDPM runtime_model`
- runtime worker
- cache

训练：

- `q_proj`
- `slot_proj`
- query embeddings
- cross-attn / FFN
- all output heads
- `region15 -> chunk_len` adapter

Qwen 是否更新：

- 若 `trainer.freeze_modules=qwen_vl_interface`：
  - 只训练 fusion head
- 若不冻结：
  - 梯度可经 `pooled_hidden` 回到 Qwen

### 5.6 复杂度结论

在当前主链路中：

- A-head 额外计算不是主瓶颈
- runtime_model miss 才是主瓶颈

因此，推荐优先选择“结构更对、工程更稳”的 query decoder，而不是极限追求更低 FLOPs 的 gated MLP 近似。

---

## 6) 与当前代码的精确对接改造点

### 6.1 `QwenPI.py`

需要改的函数：

- `__init__`
- `_compute_optional_hook_outputs`

建议新增：

- `self.a_fusion_head`
- `self.a_head_mode`

改造原则：

- 保留现有 `a_risk_head / a_trigger_head / corrective_delta_head / corrective_region_head / a_embed_head`
- 仅在新模式下实例化 fusion head
- 旧 `lite/standalone` 行为不得回归

### 6.2 `a_module_interface.py`

需要改的部分：

- `AModulePredictions`
- `StandaloneAModuleInterface`
- `build_a_module_interface`

建议新增：

- `AModuleRuntimeBatch`
- `collect_runtime_batch(...)`
- `FusionAModuleInterface` 或在 `StandaloneAModuleInterface` 上增加 feature-collect API

关键要求：

- 复用现有 `_extract_source(...)`
- 复用现有 runtime/cache/audit 逻辑
- 不允许静默 fallback 冒充 fusion 成功

### 6.3 `optional_loss_utils.py`

原则：

- 不改 target 语义
- 继续让：
  - `risk` 从 `pseudo_labels.risk_score`
  - `trigger` 从 `pseudo_labels.trigger_label`
  - `delta` 从 `pseudo_labels.delta_action_norm`
  - `region` 从 `pseudo_labels.correction_mask / affected_region_prior`
  - `embed` 优先从 `a_outputs.dynamic_embedding`

这意味着：

- Phase3.0R 的 A-head 只负责“如何预测”
- target builder 仍按当前 frozen 语义工作

### 6.4 `train_starvla.py`

原则：

- 不改 loss key 映射
- 继续复用现有：
  - `loss/risk`
  - `loss/trigger`
  - `loss/delta`
  - `loss/region`
  - `loss/embed`

当前 trainer 的 hook loss 汇总逻辑已经足够。

### 6.5 建议新增文件

- `/Users/bazinga/code/my-starvla-v2__postmerge_validate_20260410/starVLA/model/framework/a_fusion_heads.py`

建议新增类：

- `AModuleRuntimeBatch`
- `Region15ToChunkAdapter`
- `CrossAttentionFusionAHead`
- `TinyEncoderFusionAHead`
- `GatedFusionAHead`
- `PerceiverFusionAHead`

---

## 7) 最小实验设计与门禁映射

## 7.1 Train-50

产物：

- `train_50/metrics_key_summary.json`

必须满足：

- `all_loss_finite=true`
- `loss/risk|trigger|delta|region|embed` 全 present
- `max(loss/delta) > 0`
- `max(loss/embed) > 0`
- `debug/a_module_mode_fusion=1.0`

## 7.2 Train-500

产物：

- `train_500/metrics_key_summary.json`
- `a_loss_train_nonzero_assert.json`
- `freeze_assert.json`
- `noise_assert.json`
- `real_inloop_audit.json`

必须满足：

- `all_loss_finite=true`
- `nonzero_ratio_loss_delta=1.0`
- `nonzero_ratio_loss_embed=1.0`
- `freeze_assert.json.gate_pass=true`
- `noise_assert.json.gate_pass=true`
- `vdpm_model_calls_total > 0`
- `fallback_to_precomputed_count = 0`

## 7.3 Train-2000 / Train-5000

产物：

- `train_2000/metrics_key_summary.json`
- `train_5000/metrics_key_summary.json`
- longrun assert jsons

门禁：

- `all_loss_finite=true`
- `loss/delta` 非零比例 `>= 0.95`
- `loss/embed` 非零比例 `>= 0.95`
- `freeze/noise/inloop` gate 全 pass
- 无 `nan|traceback|runtimeerror`

## 7.4 Eval smoke

产物：

- `eval_smoke_summary.json`

门禁：

- `pass=true`
- `blocked=false`
- `traceback/valueerror/runtimeerror=0`

### 7.5 `not_worse_than_029` 的建议定义

Phase3.0R 不建议把 raw loss 数值和 029 做硬比较。

建议继续沿用 029 的 gate-vector 判据，定义：

- `train_50_gate_pass=true`
- `train_500_gate_pass=true`
- `candidate_train_50_all_loss_finite=true`
- `candidate_train_500_all_loss_finite=true`
- `nonzero_ratio_loss_delta=1.0`
- `nonzero_ratio_loss_embed=1.0`
- `vdpm_model_calls_total > 0`
- `fallback_to_precomputed_count = 0`
- `eval_smoke.pass=true`
- `eval_smoke` 无错误扫描命中
- 额外要求：
  - `debug/a_module_mode_fusion=1.0`

029 当前参考值：

- `max_loss_delta=0.20703125`
- `max_loss_embed=0.97265625`
- `vdpm_model_calls_total=21`
- `cache_hit_ratio=0.997375`
- `not_worse_than_lookup028=true`

说明：

- 这些值应作为 side-by-side 参考，不应作为硬 gate block。

---

## 8) 风险与回滚策略

### 8.1 主要风险

1. `region=15` 与 `chunk_len=8` 的双口径处理不清导致 shape/语义漂移
2. 把 frozen A scalar 重新作为输入导致模型学成复制器
3. 新模式侵入旧 `lite/standalone`
4. runtime fallback 被误当成 fusion 成功
5. optimizer param group 未显式覆盖新 fusion module，导致 silently 落回 base lr

### 8.2 回滚策略

回滚必须通过 mode 开关完成，不通过删代码完成：

- `framework.a_module.mode=standalone`
- `framework.a_module.mode=lite`

要求：

- 新增 `fusion` 模式失败时，旧模式仍能无改动运行
- 不修改受保护文件：
  - `starVLA/config/training/starvla_train_pi_qwen25.yaml`
  - `docs/starvla_retrofit/handoff/p4_1_qwen25_longrun_authoritative_baseline.md`

---

## 9) 最终实施建议

推荐实施顺序：

1. 先只实现 `CrossAttentionFusionAHead`
2. 只把 `dynamic_embedding(16)` 作为 frozen VDPM 输入
3. 显式引入 `region15 -> chunk_len` adapter
4. 保留旧 mode 不动
5. 用 CLI override 启用 fusion
6. 先跑 `train_50`
7. 再跑 `train_500`
8. 通过后再跑 `train_2000/5000 + eval smoke`

保底：

- 若 query decoder 在 `train_50/500` 就明显不稳，再退到 `Tiny Transformer Encoder Fusion A-head`

---

## 10) 一手参考

- Transformer:
  - https://arxiv.org/abs/1706.03762
- FiLM:
  - https://arxiv.org/abs/1709.07871
- Gated Multimodal Units:
  - https://arxiv.org/abs/1702.01992
- Perceiver IO:
  - https://arxiv.org/abs/2107.14795
- BLIP-2:
  - https://arxiv.org/abs/2301.12597
- VDPM official repo:
  - https://github.com/eldar/vdpm

