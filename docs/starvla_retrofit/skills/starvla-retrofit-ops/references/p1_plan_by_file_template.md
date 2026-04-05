# P1 Plan Template (By File)

For each file, provide:

- file path
- purpose
- minimal change set
- expected output schema change
- validation command
- rollback strategy

Example row:

- file path: `starVLA/dataset_builder/build_correction_dataset.py`
- purpose: generate shared pseudo-label samples
- minimal change set: add episode replay loop + sample emit
- expected output schema change: adds `delta_chunk_target`, `trigger_label`
- validation command: `python ... --dry-run --max-episodes 2`
- rollback strategy: feature flag `dataset_builder.enabled=false`
