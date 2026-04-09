# Handoff Docs Index

Read in this order:

1. `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/mask_diagnosis_full_history.md`
2. `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/system_prompt_operating_contract.md`
3. `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/remote_training_workflow.md`
4. `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/experiment_id_and_naming_convention.md`
5. `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/new_chat_bootstrap_command.md`
6. `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md`

Operational rule:

- After any major change, append one entry to:
  - `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md`
- For every new session, start from:
  - `/Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/new_chat_bootstrap_command.md`
- After module confirmation, the model should create EXP_ID entry itself (manual fallback only if execution is blocked).

Helper tools:

- `/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh`
- `/Users/bazinga/code/my-starvla-v2/tools/handoff/new_progress_entry.py`
- `/Users/bazinga/code/my-starvla-v2/tools/handoff/stamp_run_id_with_exp.py`
- `/Users/bazinga/code/my-starvla-v2/tools/fetch_latest_run_files.sh`
