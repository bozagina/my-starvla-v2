# CP-AH-4 LIBERO Evaluation Results

**Date:** 2026-04-15 ~ 2026-04-16
**Evaluator:** CP-BUILD
**Task Suite:** libero_goal (10 tasks × 20 trials = 200 episodes)
**Eval Config:** seed=7, use_state=True, expected_state_dim=0 (model-side truncation)

---

## Summary


| Model                   | Training                                         | Inference Variant             | Success Rate        | vs 30k Base | vs Baseline |
| ----------------------- | ------------------------------------------------ | ----------------------------- | ------------------- | ----------- | ----------- |
| **30k Base (old code)** | 30k ckpt, original code                          | Old code eval                 | **57.5%** (115/200) | Reference   | —           |
| **40k (old code)**      | 30k ckpt + 10k steps, original code              | Old code eval                 | **0.0%** (0/200)    | -57.5pp     | —           |
| **Baseline (33k)**      | 30k ckpt + A-module only, 3000 steps             | Standard                      | **69.0%** (138/200) | **+11.5pp** | Reference   |
| **CF (33k) — original** | 30k ckpt + A-module + CF, 3000 steps             | Original (buggy a_base)       | **62.0%** (124/200) | **+4.5pp**  | -7.0pp      |
| **CF (33k) — Plan A**   | same as above                                    | CF inference DISABLED (t=1.1) | **62.0%** (124/200) | **+4.5pp**  | -7.0pp      |
| **CF (33k) — Plan C**   | same as above                                    | Fixed: _cf_prev_chunk + region_gate | **64.0%** (128/200) | **+6.5pp**  | -5.0pp      |


### CP-AH-4 Acceptance Criterion

> CF model success rate ≥ Baseline - 2pp → CF ≥ 67.0%

**Verdict: FAIL** — Best CF variant (Plan C) at 64.0% is 5.0pp below Baseline, exceeding the -2pp tolerance.

### Key Finding: All New-Code Models Significantly Outperform 30k Base

The 30k Base (original code) achieves only **57.5%** on libero_goal. All new-code variants surpass it:
- **Baseline (33k): +11.5pp** — A-module training on new code substantially improves the base policy
- **Plan C (CF fixed): +6.5pp** — Even with CF training damage, the model still outperforms the original 30k
- **40k checkpoint: 0%** — catastrophic failure regardless of code version (old code also produces 0%)

---

## Per-Task Breakdown (Full Comparison)


| Task | 30k Base (old) | 40k (old) | Baseline (33k) | CF original | Plan A (CF禁用) | Plan C (修复推理) |
| ---- | -------------- | --------- | -------------- | ----------- | --------------- | ----------------- |
| 0    | 15%            | 0%        | 60%            | 85%         | 80%             | **95%**           |
| 1    | 80%            | 0%        | 100%           | 90%         | 100%            | 95%               |
| 2    | 90%            | 0%        | 95%            | 55%         | 50%             | 50%               |
| 3    | 65%            | 0%        | 35%            | 30%         | 25%             | 25%               |
| 4    | 100%           | 0%        | 100%           | 100%        | 95%             | 100%              |
| 5    | 5%             | 0%        | 45%            | 15%         | 25%             | 30%               |
| 6    | 55%            | 0%        | 50%            | 60%         | 65%             | 55%               |
| 7    | 75%            | 0%        | 100%           | 90%         | 90%             | 90%               |
| 8    | 90%            | 0%        | 100%           | 90%         | 85%             | 85%               |
| 9    | 0%             | 0%        | 5%             | 5%          | 5%              | 15%               |
| **∑** | **57.5%**     | **0.0%**  | **69.0%**      | **62.0%**   | **62.0%**       | **64.0%**         |


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

## 30k / 40k Old-Code Baselines

### 30k Base (old code): 57.5%
- Evaluated with original codebase on `evaluate-2dvlm` branch, conda env `llava3d_vla_train`
- State truncated from 8→7 dims via monkey-patch wrapper (model trained with state_dim=7)
- This establishes the true pre-training baseline for all our experiments

### 40k (old code): 0.0% — Catastrophic Failure
- **New code eval**: 0% with NaN/Inf (code incompatibility, `strict=False` loading)
- **Old code eval**: 0% with NO errors — model runs but produces completely wrong actions
- The 40k checkpoint has catastrophically forgotten the libero_goal task distribution
- This confirms continued training beyond 30k on the original codebase was counterproductive

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


| Artifact                    | Path                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| Baseline checkpoint         | `results/Checkpoints/cp_ah4_baseline_3k_20260415_081523/checkpoints/steps_33000_pytorch_model.pt` |
| CF checkpoint               | `results/Checkpoints/cp_ah4_cf_3k_20260415_104250/checkpoints/steps_33000_pytorch_model.pt`       |
| Baseline eval logs          | `results/LiberoEval/cp_ah4_baseline_eval/`                                                        |
| CF eval logs (original)     | `results/LiberoEval/cp_ah4_cf_eval/`                                                              |
| Plan A eval logs            | `results/LiberoEval/cp_plan_a_cf_no_correction/`                                                  |
| Plan C eval logs            | `results/LiberoEval/cp_plan_c_fixed_cf/`                                                          |
| 30k old-code eval logs      | `(old repo) results/LiberoEval/old_code_30k_base/`                                                |
| 40k old-code eval logs      | `(old repo) results/LiberoEval/old_code_40k_ablation/`                                            |
| 40k new-code eval logs      | `results/LiberoEval/cp_ah4_40k_eval/`                                                             |


