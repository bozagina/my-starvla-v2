# System Prompt Operating Contract (Project-Specific)

Use this as the persistent operating contract for any model that continues this project.

---

## 1) Mission

Maintain and improve the mask-generation + causal-feedback pipeline with two simultaneous goals:

1. Keep or improve task performance.
2. Increase mask selectivity/interpretability (less flat, more task-relevant).

Primary context (read first):

- `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/mask_diagnosis_full_history.md`
- `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md`
- `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/remote_training_workflow.md`
- `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/experiment_id_and_naming_convention.md`
- `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/new_chat_bootstrap_command.md`

---

## 2) Startup Protocol for New Model Sessions (Mandatory)

When a new model takes over:

1. Read the three handoff docs above.
2. Summarize understanding in <= 8 bullets.
3. Ask the user one focused question before coding:
   - "请确认你希望我本轮负责哪个模块（例如 mask 生成、feedback loss、训练配置、诊断脚本）？"
4. After user confirms ownership module:
   - perform deep analysis for that module only,
   - propose minimal plan,
   - then start implementation.
5. Create an EXP_ID progress entry before first code patch:
   - `python /Users/bazinga/code/my-starvla-v2/tools/handoff/new_progress_entry.py --module <MODULE> --owner OC --title "<task title>"`
   - or use wrapper: `/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start --module <MODULE> --owner OC --title "<task title>"`
6. The model must run this command itself after module confirmation.
   - Do not ask the user to run the progress-entry command unless tool execution is unavailable.

Do not start broad multi-module edits before module ownership is confirmed by user.

---

## 3) Ground Truth from Prior Work (Do Not Re-litigate Without New Evidence)

1. LLaVA vision fallback exists in current path.
2. Replacing visual source with official CLIP tower + current mm_projector produced near-identical alpha.
3. Official checkpoint branch (with image-only compatibility patch) also produced near-identical alpha.
4. Current dominant hypothesis: weak/indirect supervision, not tower mismatch.

Therefore:

- Do not spend primary effort repeatedly proving tower mismatch unless contradictory evidence appears.
- Prioritize supervision/objective design and measurable mask quality gains.

---

## 4) Historical Experiment Snapshot (Must Be Known)

Reference run outcomes (single-sample diagnostic):

- current:
  - `alpha_entropy = 5.5428238`
  - `alpha_max_mean = 0.00506717`
  - `topk_mass_32 = 0.1417938`
- llava_rebuild:
  - `alpha_entropy = 5.5436287`
  - `alpha_max_mean = 0.00469321`
  - `topk_mass_32 = 0.1374233`
- official (patched image-only):
  - `alpha_entropy = 5.5430450`
  - `alpha_max_mean = 0.00451871`
  - `topk_mass_32 = 0.1408636`

Similarity:

- `current_vs_llava_alpha_diff.alpha_cosine_mean = 0.995620`
- `current_vs_official_alpha_diff.alpha_cosine_mean = 0.995694`

Interpretation baseline:

- Mask is flat-ish but not strict uniform.
- Tower replacement alone is unlikely to be the breakthrough direction.

---

## 5) Non-Negotiable Working Rules

1. One hypothesis per patch set when possible.
2. Preserve behavior outside target module.
3. No silent config drift; every hyperparameter change must include rationale.
4. Evidence first, intuition second.
5. Every major modification must be appended to:
   - `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md`
6. After major code changes, always provide a suggested commit message in this format:
   - `[<EXP_ID>] <one-line intent>`
7. Respect local-vs-remote boundary:
   - local edits are not live on remote training servers until user push/pull + relaunch.
   - do not claim remote training has applied local changes without explicit evidence.
8. If training config YAML is modified, stamp run_id with EXP_ID before handoff:
   - `python /Users/bazinga/code/my-starvla-v2/tools/handoff/stamp_run_id_with_exp.py --config <TRAIN_CONFIG_PATH> --exp-id <EXP_ID> --in-place`

Major modification includes:

- algorithm change,
- loss/weight schedule change,
- data processing or feature path change,
- diagnostic script logic change,
- any decision that changes project strategy.

---

## 6) Required Metrics for Any Mask-Related Change

At minimum report:

- `soft_mask_entropy`
- `soft_mask_topk_mass_32`
- `soft_mask_alpha_max_mean`
- `soft_mask_attn_top1_mean_vis`
- `feedback_mask_contrast_*` (active/term/weighted/raw_diff)
- `delta_action_*` (gate/alpha/effective_norm/clip_saturation)

If running comparison script, also report:

- `current_vs_llava_alpha_diff` (cosine/l1/l2)
- `current_vs_official_alpha_diff` (cosine/l1/l2)

---

## 7) Development Loop (Mandatory)

For each task cycle:

1. Read latest progress entries.
2. State one-sentence hypothesis.
3. Make smallest viable patch.
4. Run targeted validation.
5. Decide keep/rollback.
6. Append progress entry with evidence.

---

## 8) Priority Ladder for This Project

P0:

- Strengthen direct supervision for mask quality.
- Build batch-level diagnostics (not only one sample).

P1:

- Try structured mask generators (e.g., directed masked self-attn over [L,V,G]).
- Improve contrast/sparsity objectives with stable ramps.

P2:

- Refactor diagnostics into reusable module.
- Keep official-compat patch code isolated in tooling.

---

## 9) Safety / Hygiene

1. Never use destructive git operations.
2. Never discard unrelated user changes.
3. Keep experiment-only patches out of core training unless intentional.
4. Mark assumptions explicitly when evidence is incomplete.

---

## 10) Remote Training Boundary (Mandatory)

1. This project develops locally but trains remotely.
2. User manually performs:
   - push local code/config to remote
   - remote pull
   - remote training launch
3. Training logs/metrics are fetched back via tools (for example:
   - `/Users/bazinga/code/my-starvla-v2/tools/fetch_latest_run_files.sh`)
4. Model should analyze remote training state only from provided logs/config/metrics.
5. If run-to-change mapping is ambiguous, ask for EXP_ID or fetched `run_identity.txt`.

---

## 11) Output Standard for Every Delivered Step

Always provide:

1. What changed (files + intent)
2. What evidence was collected
3. What conclusion is supported
4. What next action is recommended

If blocked, also provide:

- exact blocker,
- why it blocks progress,
- minimal unblock requirement.

---

## 12) Copy-Paste Prompt Block (Operational)

```text
You are continuing development for my-starvla mask/feedback pipeline.
Read first:
1) /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/mask_diagnosis_full_history.md
2) /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md
3) /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/remote_training_workflow.md
4) /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/experiment_id_and_naming_convention.md

Before coding, ask me to confirm which single module you should own this round.
After I confirm, run the EXP_ID progress-entry command yourself, report EXP_ID, then analyze and implement.

Hard constraints:
- Prior evidence indicates tower mismatch is not the primary root cause.
- Focus on stronger/direct mask supervision and measurable mask selectivity improvements.
- Report metrics: entropy, topk_mass_32, alpha_max_mean, contrast terms, delta_action terms.
- Every major modification must be appended to progress_live.md.
- Keep patches minimal, reproducible, and non-destructive.
- Do not ask me to manually run the EXP_ID logging command unless your tool execution is blocked.
- After each major code change, return a suggested commit message: [EXP_ID] one-line intent.
```
