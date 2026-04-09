# Experiment ID and Naming Convention

Purpose: avoid confusion when multiple models/operators run in parallel.

---

## 1) Core Principle

Every meaningful experiment/change must have a unique Experiment ID (`EXP_ID`).
The same `EXP_ID` must appear in:

1. progress log entry title,
2. output json / report filename,
3. optional run_id suffix,
4. commit message (if committed).

---

## 2) EXP_ID Format (Mandatory)

Use:

`ALG1-<MODULE>-<YYYYMMDD>-<SEQ>-<OWNER>`

Where:

- `ALG1`: fixed project prefix for this stream.
- `<MODULE>`: one of:
  - `MASK`
  - `FBLOSS`
  - `CFG`
  - `DIAG`
  - `DATA`
  - `INFRA`
- `<YYYYMMDD>`: local date, e.g. `20260301`.
- `<SEQ>`: 3-digit sequence per day/module, e.g. `001`, `002`.
- `<OWNER>`: short operator/model tag, e.g. `OC` (OpenCode), `GPTX`, `HUMAN`.

Example:

- `ALG1-MASK-20260301-003-OC`

---

## 3) Progress Log Naming

In `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md`:

- Entry title must start with EXP_ID.
- Recommended form:
  - `## [YYYY-MM-DD HH:MM:SS +08:00] <EXP_ID> <Short Title>`

Example:

- `## [2026-03-01 18:20:10 +08:00] ALG1-MASK-20260301-003-OC Increase mask contrast weight`

---

## 4) Output Artifact Naming

### 4.1 JSON / diagnostics

Pattern:

- `<exp_id>__<purpose>.json`

Example:

- `ALG1-DIAG-20260301-002-OC__mask_vision_compare.json`

### 4.2 Local report markdown

Pattern:

- `<exp_id>__report.md`

---

## 5) Training Run Naming (run_id)

If changing training config, append experiment identity suffix:

- preferred: `<base_run_id>__<EXP_ID>`
- fallback: `<base_run_id>__<exp_id_short>`

Where:

- `exp_id_short = <MODULE>-<MMDD>-<SEQ>`

Examples:

- `mapanything_llava3dact_vla_ab_b_concat_cross_v4_1_1__ALG1-MASK-20260301-003-OC`
- `mapanything_llava3dact_vla_ab_b_concat_cross_v4_1_1__MASK-0301-003` (fallback)

Do not overwrite previous run_id for distinct experiments.

Recommended helper:

```bash
python /Users/bazinga/code/my-starvla-v2/tools/handoff/stamp_run_id_with_exp.py \
  --config <training_yaml> \
  --exp-id <EXP_ID> \
  --in-place
```

---

## 6) Branch / Commit Naming (Optional but Recommended)

### 6.1 Branch

- `codex/<exp_id-lower>`

Example:

- `codex/alg1-mask-20260301-003-oc`

### 6.2 Commit message

- `[<EXP_ID>] <one-line intent>`

Example:

- `[ALG1-MASK-20260301-003-OC] add directed self-attn mask prototype`

---

## 7) Status Taxonomy (Use Consistently)

Allowed status values in progress entries:

- `IN_PROGRESS`
- `DONE`
- `BLOCKED`
- `ROLLED_BACK`

If `ROLLED_BACK`, include rollback reason and which metrics regressed.

---

## 8) Metric Snapshot Key (for comparability)

For each experiment, include these fields in report/progress:

- mask:
  - `soft_mask_entropy`
  - `soft_mask_topk_mass_32`
  - `soft_mask_alpha_max_mean`
  - `soft_mask_attn_top1_mean_vis`
- feedback:
  - `feedback_mask_contrast_term`
  - `feedback_mask_contrast_weighted`
  - `delta_action_alpha`
  - `delta_action_effective_norm_mean`
- optional compare:
  - `current_vs_llava_alpha_diff.alpha_cosine_mean`
  - `current_vs_official_alpha_diff.alpha_cosine_mean`

---

## 9) Collision Prevention Rules

1. Before creating new EXP_ID, check latest progress entry sequence.
2. If two operators generate same sequence, later one increments by +1 and records correction.
3. Never reuse an EXP_ID, even if experiment failed.
4. Failed experiments are still valid IDs and must remain traceable.

---

## 10) Remote Log Mapping Rule

When training logs are fetched to local:

1. Use `/Users/bazinga/code/my-starvla-v2/tools/fetch_latest_run_files.sh`.
2. Check generated `<run_dir>/run_identity.txt`.
3. Use `exp_id` in that file to map remote run -> local code change -> progress entry.

If `exp_id` is empty, the run is considered weakly traceable and should be marked in analysis.

---

## 11) Quick Start Checklist

Before starting work:

1. Pick module and create EXP_ID.
2. Add IN_PROGRESS entry in progress log.
3. Use EXP_ID in artifact names.
4. On completion, update entry to DONE/BLOCKED/ROLLED_BACK with metrics.

Helper command:

```bash
python /Users/bazinga/code/my-starvla-v2/tools/handoff/new_progress_entry.py \
  --module MASK \
  --owner OC \
  --title "Short task title"
```
