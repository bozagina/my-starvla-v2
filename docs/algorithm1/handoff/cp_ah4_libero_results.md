# CP-AH-4 LIBERO Evaluation Results

**Date:** 2026-04-15 ~ 2026-04-16
**Evaluator:** CP-BUILD
**Task Suite:** libero_goal (10 tasks × 20 trials = 200 episodes)
**Eval Config:** seed=7, use_state=True, expected_state_dim=0 (model-side truncation)

---

## Summary


| Model                   | Training                                         | Inference Variant             | Success Rate        | Result             |
| ----------------------- | ------------------------------------------------ | ----------------------------- | ------------------- | ------------------ |
| **Baseline (33k)**      | 30k ckpt + A-module only, 3000 steps             | Standard                      | **69.0%** (138/200) | Reference          |
| **CF (33k) — original** | 30k ckpt + A-module + CF, 3000 steps             | Original (buggy a_base)       | **62.0%** (124/200) | -7.0pp vs Baseline |
| **CF (33k) — Plan A**   | same as above                                    | CF inference DISABLED (t=1.1) | **62.0%** (124/200) | -7.0pp vs Baseline |
| **CF (33k) — Plan C**   | same as above                                    | Fixed: _cf_prev_chunk + region_gate | **64.0%** (128/200) | -5.0pp vs Baseline |
| **40k ablation**        | 30k ckpt + 10k steps (old code)                  | N/A                           | **0.0%** (0/200)    | Incompatible (NaN) |


### CP-AH-4 Acceptance Criterion

> CF model success rate ≥ Baseline - 2pp → CF ≥ 67.0%

**Verdict: FAIL** — Best CF variant (Plan C) at 64.0% is 5.0pp below Baseline, exceeding the -2pp tolerance.

---

## Per-Task Breakdown (Full Comparison)


| Task | Baseline | CF original | Plan A (CF禁用) | Plan C (修复推理) | Plan C vs Baseline |
| ---- | -------- | ----------- | --------------- | ----------------- | ------------------ |
| 0    | 60%      | 85%         | 80%             | **95%**           | **+35pp**          |
| 1    | 100%     | 90%         | 100%            | **95%**           | -5pp               |
| 2    | 95%      | 55%         | 50%             | **50%**           | **-45pp**          |
| 3    | 35%      | 30%         | 25%             | **25%**           | -10pp              |
| 4    | 100%     | 100%        | 95%             | **100%**          | 0pp                |
| 5    | 45%      | 15%         | 25%             | **30%**           | -15pp              |
| 6    | 50%      | 60%         | 65%             | **55%**           | +5pp               |
| 7    | 100%     | 90%         | 90%             | **90%**           | -10pp              |
| 8    | 100%     | 90%         | 85%             | **85%**           | -15pp              |
| 9    | 5%       | 5%          | 5%              | **15%**           | +10pp              |
| **∑** | **69.0%** | **62.0%** | **62.0%**       | **64.0%**         | **-5.0pp**         |


### Key Observations

1. **CF training damages base model**: Plan A (CF inference disabled) = 62%, same as original CF → the -7pp deficit comes from training, not inference
2. **Plan C inference fix adds +2pp**: 64% vs 62%, showing corrective flow has positive signal when the inference path is fixed
3. **Fix helps dramatically on T0**: 95% vs 80% (Plan A) → +15pp from corrective flow inference alone
4. **Training damage is task-specific**: T2 (-45pp), T5 (-15pp), T8 (-15pp) are severely hurt by CF training, not recoverable by inference fix
5. **T9 surprise improvement**: Plan C achieves 15% on T9 vs 5% for all other variants
6. **Incomplete fix**: Client-side `reset()` does NOT send reset signal to server, so `_cf_prev_chunk` leaks across episodes

---

## Ablation Analysis (CP-RES Recommendations)

### Plan A: 推理消融（CF inference disabled via trigger_threshold=1.1）

- **Purpose**: Determine if CF training damages the base model
- **Result**: 62.0% = identical to original CF with inference enabled
- **Conclusion**: CF training itself causes -7pp degradation; original CF inference contributes 0pp net (positive on some tasks, negative on others, cancels out)

### Plan C: 推理修复（_cf_prev_chunk + region_gate thresholding）

- **Fixes applied**:
  1. `QwenPI.predict_action()`: Use `_cf_prev_chunk` (previous step's base prediction) instead of current `pred_actions` as input to CF head
  2. `corrective_flow_head.predict()`: Apply `region_gate = clamp(sigmoid(region_logits) - 0.3, min=0)` threshold
  3. Server-side `reset` RPC handler added to `websocket_policy_server.py`
- **NOT applied**: Client-side `model2libero_interface.py` does not send `reset` to server → cross-episode `_cf_prev_chunk` contamination
- **Result**: +2pp over original CF (64% vs 62%), showing corrective flow has positive signal
- **Conclusion**: Inference fix partially effective; full potential masked by (a) training damage to base model, (b) missing client-side reset

### Root Cause Decomposition

| Factor | Impact | Status |
| ------ | ------ | ------ |
| CF training → base model damage | **-7pp** (dominant) | Needs training-level fix (loss weight, gradient isolation) |
| Inference: a_base semantic mismatch | ~-2pp | Fixed in Plan C |
| Inference: region_gate unthresholded | minor | Fixed in Plan C |
| Inference: cross-episode state leak | unknown (est. -1~2pp) | NOT fixed (client reset missing) |

---

## 40k Ablation Baseline (Incompatible)

The 40k checkpoint (`ablation_baseline_QwenPI_s42_20260311_211605/steps_40000`) was trained with the **old codebase** and loaded into the new code via `strict=False` (30 missing keys: A-module + corrective heads). The model produced **NaN/Inf actions** (21+ MuJoCo instability warnings), resulting in 0% success rate. This is a code compatibility issue, not a model quality issue.

---

## Training Metrics (Final)

### Baseline (33k)

- Checkpoint: `cp_ah4_baseline_3k_20260415_081523/steps_33000`
- loss/action final: ~0.046
- loss/a_module final: ~0.001

### CF (33k)

- Checkpoint: `cp_ah4_cf_3k_20260415_104250/steps_33000`
- loss/action final: 0.047
- loss/corrective_flow final: 0.008
- loss/corrective_flow_base: 0.007
- loss/corrective_flow_region: 0.0003
- loss/a_module final: 0.001

---

## Artifact Paths


| Artifact               | Path                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| Baseline checkpoint    | `results/Checkpoints/cp_ah4_baseline_3k_20260415_081523/checkpoints/steps_33000_pytorch_model.pt` |
| CF checkpoint          | `results/Checkpoints/cp_ah4_cf_3k_20260415_104250/checkpoints/steps_33000_pytorch_model.pt`       |
| Baseline eval logs     | `results/LiberoEval/cp_ah4_baseline_eval/`                                                        |
| CF eval logs (original)| `results/LiberoEval/cp_ah4_cf_eval/`                                                              |
| Plan A eval logs       | `results/LiberoEval/cp_plan_a_cf_no_correction/`                                                  |
| Plan C eval logs       | `results/LiberoEval/cp_plan_c_fixed_cf/`                                                          |
| 40k eval logs          | `results/LiberoEval/cp_ah4_40k_eval/`                                                             |


