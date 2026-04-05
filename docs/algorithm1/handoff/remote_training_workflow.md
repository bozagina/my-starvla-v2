# Local Dev / Remote Training Workflow

This project uses split execution:

1. Code development and review are done locally.
2. Training jobs run on remote server(s).

---

## 1) Responsibilities Split

- Local side:
  - code edits
  - config edits
  - diagnostics and documentation updates
- Remote side:
  - actual training launch
  - long-running logs/metrics generation

Because of this split:

- The model must not assume local changes are already active on server.
- User performs push/pull + server-side launch manually.

---

## 2) Required Handoff Data for Analysis

When asking model to analyze training status, provide:

1. fetched run artifacts (`config.yaml`, `metrics.jsonl`, `summary.jsonl`, `train.log`)
2. corresponding EXP_ID (or run_id containing EXP tag)
3. relevant local code/config diffs

Fetch helper:

- `/Users/bazinga/code/my-starvla-v2/tools/fetch_latest_run_files.sh`

The fetch script now writes:

- `<run_dir>/run_identity.txt`

including:

- `run_id`
- parsed `exp_id` (if present in run_id)
- parsed short run tag

---

## 3) Training-Run Identity Requirement

To map logs back to specific code changes, training run must include experiment identity in `run_id`.

Recommended format:

- `...__ALG1-<MODULE>-<YYYYMMDD>-<SEQ>-<OWNER>`

Helper script:

```bash
python /Users/bazinga/code/my-starvla-v2/tools/handoff/stamp_run_id_with_exp.py \
  --config <training_yaml> \
  --exp-id <EXP_ID> \
  --in-place
```

This makes downstream log-to-change mapping deterministic.

---

## 4) Commit Message Requirement

After each major code change, model should return suggested commit message:

- `[<EXP_ID>] <one-line intent>`

This must also be captured in progress log entries for traceability.

---

## 5) Typical End-to-End Sequence

1. New chat starts; model reads handoff docs and asks module ownership.
2. User confirms module.
3. Model creates EXP_ID progress entry.
4. Model edits code and returns:
   - what changed
   - evidence
   - suggested commit message
5. User pushes to remote and triggers training.
6. User fetches latest logs via fetch script.
7. User sends logs/config/metrics back to model.
8. Model updates analysis + progress history.
