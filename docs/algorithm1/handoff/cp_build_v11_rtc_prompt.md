# CP-BUILD 执行 Prompt — RFC v1.1 RTC 全流程

```
thread_id:  CP-BUILD
round_id:   CP-ROUND-FUSION-RESUME
gate:       phase3_0b_cp_fusion_build
upstream:   docs/algorithm1/handoff/cp_res_policy_rfc_v1_fusion.md (v1.1)
date:       2026-04-17
```

---

## 你的身份

你是 `CP-BUILD` 线程。你的任务是基于 CP-RES RFC v1.1（RTC 范式 + fusion mode）
完成 CorrectiveFlowHead 的架构修订、工程修复、多 seed 训练和 LIBERO 评估，
直到所有 CP-AH 验收假设有远程证据，可交付 CP-REVIEW 审核。

---

## 必读文档（按顺序）

1. `docs/algorithm1/handoff/cp_res_policy_rfc_v1_fusion.md` — **RFC v1.1 全文，最重要**
2. `docs/algorithm1/handoff/cp_res_interface_assumptions_v1_fusion.md` — 接口约束
3. `docs/algorithm1/handoff/cp_res_blocked_items_v1_fusion.md` — 工程任务和阻塞项
4. `docs/algorithm1/handoff/manifests/cp_current_round.yaml` — 当前状态
5. `docs/algorithm1/handoff/manifests/a_module_current_round.yaml` — A-module 证据（重点看 downstream_rule 和 tb4_evidence）

### 现有代码（必须先读再改）

6. `starVLA/model/framework/corrective_flow_head.py` — **需要重大修改的文件**
7. `starVLA/model/framework/QwenPI.py` — **需要修改 forward() 和 predict_action()**
8. `starVLA/training/train_starvla.py` — 可能需要微调 hook_specs
9. `starVLA/model/framework/a_module_interface.py` — **冻结，不得修改**，但需读懂 fusion mode
10. `starVLA/model/framework/a_fusion_heads.py` — **冻结，不得修改**，需读懂 CrossAttentionFusionAHead
11. `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` — DiT 实现，理解 cross_attention_dim 参数

---

## 开发环境边界

- **本地**（macOS，无 GPU）：编写代码、修改配置、py_compile 检查
- **远程**（Linux + GPU）：训练、推理、LIBERO eval。代码通过 git push/pull 同步
- 任何需要 GPU 的操作，你只能准备好代码和配置，明确告知用户"需要在服务器上执行以下命令"
- 远程结果未回传前，标记 BLOCKED_WAIT_REMOTE

---

## 执行计划（按顺序，不要跳步）

### Phase 1: 架构修订（本地）

#### Task 1.1: CorrectiveFlowHead RTC 改造

修改 `starVLA/model/framework/corrective_flow_head.py`：

**1.1a: `_encode_and_attend` — 移除 MetadataEncoder 输入**

当前代码：
```python
def _encode_and_attend(self, vl_embs, a_input, trigger_prob, region_logits):
    ...
    meta_input = torch.cat([trigger_prob, region_logits], dim=-1)
    meta_features = self.metadata_encoder(meta_input).unsqueeze(1)
    sa_embs = torch.cat([meta_features, action_features], dim=1)
    ...
    for block in self.dit.transformer_blocks:
        hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=vl_embs,  # VLM cross-attention
            temb=temb,
        )
    ...
    action_hidden = hidden_states[:, 1:, :]  # drop metadata token
```

改为：
```python
def _encode_and_attend(self, a_input, vl_embs=None):
    ...
    # 不再使用 MetadataEncoder
    sa_embs = action_features  # [B, C, hidden]，无 metadata token
    ...
    for block in self.dit.transformer_blocks:
        hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=vl_embs,  # BD-7: 传 None 则无 cross-attn
            temb=temb,
        )
    ...
    action_hidden = hidden_states  # 所有 token 都是 action token，无需 drop
```

**1.1b: `forward()` — region_logits 仅用于 loss**

当前签名：`forward(self, vl_embs, a_prev, a_gt, trigger_prob, region_logits)`

改为：
```python
def forward(self, a_prev, a_gt, region_logits, vl_embs=None):
    """
    Training forward.
    Args:
        a_prev:        [B, C, action_dim]  previous chunk (GT + noise)
        a_gt:          [B, C, action_dim]  current chunk GT
        region_logits: [B, C]             from fusion head (detached), loss weighting only
        vl_embs:       [B, S, vl_hidden_dim] optional VLM features (None if BD-7=no cross-attn)
    """
    velocity_gt = a_gt - a_prev
    pred_velocity = self._encode_and_attend(a_prev, vl_embs)

    # region_logits 仅用于 loss 加权，不输入模型
    region_weight = torch.sigmoid(region_logits).unsqueeze(-1)
    per_step_loss = (pred_velocity - velocity_gt) ** 2
    base_loss = per_step_loss.mean()
    region_loss = (region_weight * per_step_loss).mean()
    loss = base_loss + self.region_loss_weight * region_loss
    debug = {
        "corrective_flow_base_loss": base_loss.detach(),
        "corrective_flow_region_loss": region_loss.detach(),
    }
    return loss, pred_velocity, debug
```

**1.1c: `predict()` — 极简推理**

当前签名：`predict(self, vl_embs, a_base, trigger_prob, region_logits)`

改为：
```python
def predict(self, a_prev, vl_embs=None):
    """
    Inference: one-step corrective flow (RTC paradigm).
    No trigger gating, no region gating. Model self-determines correction.
    """
    pred_velocity = self._encode_and_attend(a_prev, vl_embs)
    a_corrected = a_prev + pred_velocity
    return a_corrected
```

**1.1d: `__init__` — BD-7 决策**

推荐去掉 VLM cross-attention：
```python
self.dit = DiT(
    ...
    cross_attention_dim=None,  # BD-7: 去掉 VLM cross-attention
)
```

如果你决定保留 cross-attention 作为消融对比，需要通过 config 参数控制：
```python
cross_attention_dim=vl_hidden_dim if use_cross_attention else None,
```

MetadataEncoder 模块可以保留在代码中（向后兼容），但 `_encode_and_attend` 不再调用它。

#### Task 1.2: QwenPI forward() 修改

修改 `starVLA/model/framework/QwenPI.py` 的 CF 训练路径（约 L510-549）：

**核心变化**：
1. region_logits 来自 fusion head 实时输出（ENG-1），不再调用 `a_module_interface.predict()`
2. RNG 对齐（ENG-3）
3. 新的 `corrective_flow_head.forward()` 签名

```python
if self.corrective_flow_enabled:
    cf_trainer_cfg = ...  # 现有 config 读取逻辑不变
    if _cfg_enabled(cf_trainer_cfg, "enabled", default=False):
        noise_scale = float(_cfg_get(cf_trainer_cfg, "noise_scale", 0.05))

        # a_prev 构造（不变）
        if actions.shape[1] >= 2 * self.chunk_len:
            a_prev = actions[:, : self.chunk_len, :]
        else:
            a_prev = torch.zeros_like(actions_target)

        # ENG-3: RNG 对齐 — 使用独立 Generator
        cf_rng = torch.Generator(device=a_prev.device)
        cf_rng.manual_seed(42)
        a_prev_noisy = a_prev + noise_scale * torch.randn(
            a_prev.shape, device=a_prev.device, dtype=a_prev.dtype,
            generator=cf_rng,
        )

        # ENG-1: region_logits 来自 fusion head 实时输出（仅 loss 权重）
        # 从 hook_output_dict 或 _compute_optional_hook_outputs 中获取
        # 如果 a_module_mode == "fusion"，_compute_optional_hook_outputs 已经
        # 通过 fusion head 计算了 region_logits，需要在该函数中缓存并传出
        #
        # 方案：在 _compute_optional_hook_outputs 中将 region_logits 存入
        #       output_dict["_cf_region_logits"] 或类似中间字段
        # 非 fusion mode 下回退到 a_module_interface.predict()
        if self.a_module_mode == "fusion":
            region_logits_for_cf = self._cached_fusion_region_logits.detach()
        else:
            pooled_for_cf = base_hidden.to(dtype=action_model_dtype).mean(dim=1)
            a_preds = self.a_module_interface.predict(
                pooled_hidden=pooled_for_cf, examples=examples,
                action_loss=action_loss, chunk_len=self.chunk_len,
            )
            region_logits_for_cf = a_preds.region_logits.detach()

        # BD-7: vl_embs 传 None 则无 cross-attn
        vl_embs_for_cf = (
            base_hidden.to(dtype=action_model_dtype).detach()
            if self.corrective_flow_use_cross_attention
            else None
        )

        cf_loss, _, cf_debug = self.corrective_flow_head(
            a_prev=a_prev_noisy.to(dtype=action_model_dtype),
            a_gt=actions_target.to(dtype=action_model_dtype),
            region_logits=region_logits_for_cf.to(dtype=action_model_dtype),
            vl_embs=vl_embs_for_cf,
        )
        output_dict["corrective_flow_loss"] = cf_loss
        for k, v in cf_debug.items():
            output_dict[k] = v
```

**关于 `_cached_fusion_region_logits`**：你需要在 `_compute_optional_hook_outputs` 中，
当 `a_module_mode == "fusion"` 时，将 `fusion_outputs["region_logits_train"]` 存到
`self._cached_fusion_region_logits`。这是最干净的传递方式。

#### Task 1.3: QwenPI predict_action() 修改

修改推理路径（约 L619-659），**大幅简化**：

```python
if self.corrective_flow_enabled:
    a_prev_for_cf = (
        self._cf_prev_chunk.to(device=pred_actions.device)
        if self._cf_prev_chunk is not None
        else torch.zeros_like(pred_actions)
    )

    vl_embs_for_cf = (
        base_hidden.float()
        if self.corrective_flow_use_cross_attention
        else None
    )

    with torch.autocast("cuda", enabled=False):
        orig_cf_dtype = next(self.corrective_flow_head.parameters()).dtype
        try:
            self.corrective_flow_head.float()
            a_corrected = self.corrective_flow_head.predict(
                a_prev=a_prev_for_cf.float(),
                vl_embs=vl_embs_for_cf,
            )
            normalized_actions = a_corrected.detach().to(dtype=torch.float32).cpu().numpy()
        finally:
            self.corrective_flow_head.to(dtype=orig_cf_dtype)

    self._cf_prev_chunk = pred_actions.detach().clone()
```

注意：
- 不再调用 `ami.predict()`
- 不再检查 `trigger_prob > threshold`
- 不再需要 ami 的 dtype 管理代码
- ENG-4: 确保 `reset_cf_state()` 在 eval client 的 episode 循环中被调用

#### Task 1.4: __init__ 配置读取

在 `QwenPI.__init__` 中新增 config 项（或复用现有）：

```python
# BD-7 配置
cf_cfg = _cfg_get(framework_cfg, "corrective_flow", None)
self.corrective_flow_use_cross_attention = bool(
    _cfg_get(cf_cfg, "use_cross_attention", False)  # 默认 False = self-attn only
)
```

同时修改 `CorrectiveFlowHead` 实例化：
```python
self.corrective_flow_head = CorrectiveFlowHead(
    action_dim=...,
    chunk_len=...,
    hidden_dim=...,
    vl_hidden_dim=llm_hidden_size if self.corrective_flow_use_cross_attention else 0,
    ...
)
```

当 `vl_hidden_dim=0` 时在 `__init__` 中设置 `cross_attention_dim=None`。

### Phase 2: 本地验证（本地）

#### Task 2.1: py_compile 全部修改文件

```bash
python -m py_compile starVLA/model/framework/corrective_flow_head.py
python -m py_compile starVLA/model/framework/QwenPI.py
python -m py_compile starVLA/training/train_starvla.py
```

所有文件必须 PASS。

#### Task 2.2: 训练配置 YAML

创建或更新 fusion mode CF 训练配置：

```yaml
framework:
  a_module:
    mode: fusion
    # ... fusion 相关配置
  corrective_flow:
    enabled: true
    use_cross_attention: false  # BD-7: self-attention only
    hidden_dim: 1024
    num_layers: 4
    num_heads: 16
    region_loss_weight: 0.5
    trigger_threshold: 0.5  # 保留但推理时不使用

trainer:
  optional_loss_hooks:
    corrective_flow:
      enabled: true
      key: corrective_flow_loss
      weight: 1.0
      noise_scale: 0.05
```

### Phase 3: 远程训练（远程，需用户执行）

#### Task 3.1: 代码同步

```bash
git add -A && git commit -m "[CP-BUILD] RFC v1.1 RTC: remove MetadataEncoder input, simplify CF inference, RNG alignment"
git push
# 在远程服务器：
cd /path/to/remote/repo && git pull
```

#### Task 3.2: CP-AH-5 验证 — RNG 对齐基线

**目的**：确认 RNG 对齐后，CF 训练不影响 base model 性能。

```bash
# Fusion Baseline（无 CF）
python -m starVLA.training.train_starvla \
  --config_yaml starVLA/config/training/starvla_train_pi_qwen25_fusion_baseline.yaml \
  --seed 42 \
  --run_name cp_ah5_baseline_seed42

# Fusion CF (RTC) — CF 推理禁用（仅训练）
python -m starVLA.training.train_starvla \
  --config_yaml starVLA/config/training/starvla_train_pi_qwen25_fusion_cf_rtc.yaml \
  --seed 42 \
  --run_name cp_ah5_cf_noinfer_seed42
```

评估两个模型（CF 推理均禁用）：
```bash
# eval baseline
python eval_libero.py --ckpt cp_ah5_baseline_seed42 --trials 20 --tasks 10
# eval CF model (inference disabled)
python eval_libero.py --ckpt cp_ah5_cf_noinfer_seed42 --trials 20 --tasks 10 --disable_cf
```

**通过条件**：两者 Success@LIBERO 差 < 1pp。

#### Task 3.3: 多 seed 训练 — CP-AH-4-v1

三组 seed 训练 Fusion Baseline 和 Fusion CF：

```bash
for SEED in 42 123 456; do
  # Fusion Baseline
  python -m starVLA.training.train_starvla \
    --config_yaml starvla_train_pi_qwen25_fusion_baseline.yaml \
    --seed $SEED \
    --run_name cp_ah4_baseline_seed${SEED}

  # Fusion CF (RTC) — CF 推理启用
  python -m starVLA.training.train_starvla \
    --config_yaml starvla_train_pi_qwen25_fusion_cf_rtc.yaml \
    --seed $SEED \
    --run_name cp_ah4_cf_rtc_seed${SEED}
done
```

#### Task 3.4: 多 seed 评估

```bash
for SEED in 42 123 456; do
  python eval_libero.py --ckpt cp_ah4_baseline_seed${SEED} --trials 20 --tasks 10
  python eval_libero.py --ckpt cp_ah4_cf_rtc_seed${SEED} --trials 20 --tasks 10
done
```

**通过条件**：CF 三组均值 ≥ Baseline 三组均值 - 2pp。

### Phase 4: CP-AH-1~3 验证（从远程 500-step smoke 日志中提取）

在 Task 3.2 的训练中同时检查：
- CP-AH-1: `debug/corrective_loss_region_trigger_filtered_count` 确认 P0 生效
- CP-AH-2: `metrics.jsonl` 中所有 loss 有限
- CP-AH-3: `loss/consist` 下降趋势

### Phase 5: 交付（本地）

#### Task 5.1: 产出结果文档

创建 `docs/algorithm1/handoff/cp_ah4_v11_rtc_libero_results.md`，包含：

```markdown
# CP-AH-4-v1 / CP-AH-5 LIBERO Results — RFC v1.1 RTC

## CP-AH-5: RNG 对齐验证
| Model | Seed | CF Inference | Success@LIBERO |
|-------|------|-------------|----------------|
| Baseline | 42 | N/A | xx.x% |
| CF (no infer) | 42 | disabled | xx.x% |
| **差值** | | | xx.x pp |
| **判定** | | | PASS / FAIL (< 1pp) |

## CP-AH-4-v1: 多 seed 评估
| Model | Seed 42 | Seed 123 | Seed 456 | 均值 | 标准差 |
|-------|---------|----------|----------|------|--------|
| Baseline | xx.x% | xx.x% | xx.x% | xx.x% | x.x |
| CF (RTC) | xx.x% | xx.x% | xx.x% | xx.x% | x.x |
| **差值** | | | | xx.x pp | |
| **判定** | | | | PASS / FAIL (≥ -2pp) |

## CP-AH-1~3
| # | 指标 | 结果 | 判定 |
|---|------|------|------|
| CP-AH-1 | P0 region filtering | xxx | PASS/FAIL |
| CP-AH-2 | All losses finite | xxx | PASS/FAIL |
| CP-AH-3 | consist loss trend | xxx | PASS/FAIL |
```

#### Task 5.2: 更新 manifest 和 progress_live

将 CP-BUILD 状态更新为 `DONE_PENDING_REVIEW`，附上远程证据路径和 CP-AH 判定。

#### Task 5.3: 交付给 CP-REVIEW

CP-REVIEW 需要审核：
1. 架构修订代码质量（MetadataEncoder 移除、签名变更）
2. ENG-1/3/4 修复正确性
3. CP-AH-1~5 全部远程证据
4. 冻结锚点未被违反

---

## 不得做的事

1. 不得修改 `a_module_interface.py`（冻结）
2. 不得修改 `a_fusion_heads.py`（冻结）
3. 不得修改 `optional_loss_utils.py`（冻结）
4. 不得在推理路径中调用 A-module / fusion head 给 CF 使用
5. 不得让 trigger/region 作为 CF 模型的输入特征（BD-8）
6. 不得在本地声称训练/评估通过（必须有远程证据）
7. 不得跳步——Phase 1 → 2 → 3 → 4 → 5 严格顺序执行
8. 不得把 Phase 3 的远程命令直接执行（准备好命令后通知用户）

## 必须做的事

1. 每个 Phase 完成后更新 progress_live.md
2. 远程结果未回传前标记 BLOCKED_WAIT_REMOTE
3. Phase 1~2 完成后做一次 git commit
4. Phase 3~4 远程结果回传后做一次 git commit
5. Phase 5 完成后做最终 git commit
6. 产出物必须包含：代码 diff、配置 diff、远程证据、CP-AH 判定、changed-file handoff note
