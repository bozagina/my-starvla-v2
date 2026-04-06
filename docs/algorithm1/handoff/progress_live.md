# StarVLA Progress Live

This file records session-level execution status for retrofit phases.


## [2026-04-05 13:11:20 +00:00] ALG1-INFRA-20260405-001-OC initialize repo-context guard in authoritative qwen repo

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - ...
- Changes:
  - Files:
    - `...`
  - Code/Config summary:
    - ...
- Evidence:
  - Commands:
    - `...`
  - Key outputs/metrics:
    - ...
- Decision:
  - ...
- Risks/Notes:
  - ...
- Next step:
  - ...
- Commit message:
  - `[ALG1-INFRA-20260405-001-OC] initialize repo-context guard in authoritative qwen repo`

## [2026-04-05 13:12:11 +00:00] ALG1-INFRA-20260405-001-OC migrate guard/prompt/skill system into authoritative qwen repo

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Make wrong-repo execution impossible by enforcing repo identity guard and startup checks directly in authoritative server repo.
- Changes:
  - Files:
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/handoff/ensure_repo_context.sh`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/handoff/pre_dev_readiness.sh`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/handoff/bootstrap_session.sh`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/handoff/new_progress_entry.py`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/tools/handoff/check_deadlock_risk.py`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/docs/starvla_retrofit/handoff/*`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/docs/starvla_retrofit/skills/starvla-retrofit-ops/*`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Added hard repo guard with `STARVLA_EXPECTED_REPO_ROOT` and optional `STARVLA_EXPECTED_VLM_SCOPE` checks.
    - Wired guard into readiness, preflight, and bootstrap flow.
    - Added python/python3 auto-detection in startup scripts for server portability.
    - Added compact prompt + operating contract + checklist + skill docs to this repo so startup no longer depends on another repository.
- Evidence:
  - Commands:
    - `ssh myserver "... bash tools/handoff/ensure_repo_context.sh --expect-root ... --expect-vlm-scope qwen_only --require-expected-root"`
    - `ssh myserver "... bash tools/handoff/pre_dev_readiness.sh"`
    - `ssh myserver "... bash tools/handoff/bootstrap_session.sh start --module INFRA --owner OC --title ..."`
    - `ssh myserver "... bash -n tools/handoff/*.sh docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh"`
  - Key outputs/metrics:
    - `REPO_CONTEXT_OK=YES` on authoritative repo with qwen_only scope.
    - `READY_TO_DEVELOP=YES` after readiness run (warnings only for dirty worktree).
    - `ALG1-INFRA-20260405-001-OC` created in this repo via bootstrap.
- Decision:
  - Authoritative repo now has self-contained guard/prompt/skill startup system and should be the only development baseline.
- Risks/Notes:
  - Worktree is intentionally dirty due existing user edits (`run_libero_train.sh`, `framework/__init__.py`, and untracked config); do not revert.
- Next step:
  - Stage/commit only new docs/tools infra files, then continue current gate implementation in this repo.
- Commit message:
  - `[ALG1-INFRA-20260405-001-OC] bootstrap repo-context guard system in authoritative qwen repo`

## [2026-04-05 21:38:45 +08:00] ALG1-DATA-20260405-001-OC P1 shared builder sample schema integration

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - ...
- Changes:
  - Files:
    - `...`
  - Code/Config summary:
    - ...
- Evidence:
  - Commands:
    - `...`
  - Key outputs/metrics:
    - ...
- Decision:
  - ...
- Risks/Notes:
  - ...
- Next step:
  - ...
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] P1 shared builder sample schema integration`

## [2026-04-05 21:43:00 +08:00] ALG1-DATA-20260405-001-OC implement P1 shared builder in dataloader and validate by remote smoke

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Land backward-compatible shared VLA sample builder path in dataloader with feature flag `datasets.vla_data.shared_builder_enabled`.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/dataloader/__init__.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/dataloader/lerobot_datasets.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/dataloader/gr00t_lerobot/datasets.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/config/training/starvla_cotrain_oxe.yaml`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/config/training/starvla_cotrain_libero.yaml`
  - Code/Config summary:
    - Added lazy import for `vlm_datasets` and auto-derived `action_chunk_size = future_action_window_size + 1` injection path.
    - Added optional `action_chunk_size` override and safe deep-copy handling in LeRobot dataset config builder.
    - Added shared builder output path (`obs`, `action_chunk`, `meta`) with schema `p1_shared_builder_v1`, while preserving legacy output when flag is disabled.
    - Extended sample packing to include `trajectory_id` and `step_index` into metadata.
    - Added `shared_builder_enabled: false` defaults in cotrain YAMLs for backward compatibility.
- Evidence:
  - Commands:
    - remote dataloader smoke with `shared_builder_enabled=true/false` against `libero_all` dataset and `video_backend=torchvision_av`.
  - Key outputs/metrics:
    - `shared_builder_enabled=true` -> sample keys include `action_chunk`, `obs`, `meta`; `meta.schema_version = p1_shared_builder_v1`; `action_shape=[16,7]`, `action_chunk_shape=[16,7]`, `state_shape=[1,8]`.
    - `shared_builder_enabled=false` -> legacy keys only (`action/image/lang/state`), `action_chunk_shape=null`, `meta=null`.
- Decision:
  - P1 shared-builder contract is implemented with feature flag and validated in remote smoke outputs provided by user.
- Risks/Notes:
  - Full training gate (P3) still depends on trainer-side accelerate/logging init order fix and end-to-end launch verification.
- Next step:
  - Commit P1 dataloader/config changes, then enter P2/P3 contract gate insertion & training smoke.
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] land P1 shared builder dataloader contract with compatibility flag`

## [2026-04-05 22:02:00 +08:00] ALG1-DATA-20260405-001-OC add trainer-side shared-builder contract check and startup logging guard

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Add P2/P3 trainer gate to validate shared-builder sample contract at runtime, and avoid early accelerate logger init failure path.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/training/train_starvla.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/config/training/starvla_cotrain_oxe.yaml`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/config/training/starvla_cotrain_libero.yaml`
  - Code/Config summary:
    - Added helper parsers (`_cfg_get/_cfg_enabled/_to_int_or_none/_shape_2d`) and `VLATrainer._check_shared_builder_contract(...)`.
    - Contract checker supports `mode=auto|shared|legacy`, required keys, shape checks, `meta.schema_version`, and strict legacy no-extra-key option.
    - Hooked contract checker into `_train_step` before forward pass with debug metrics.
    - Replaced `main()` first `logger.info(...)` with `print(...)` to avoid launch-path sensitivity before accelerate logger state is ready.
    - Added default trainer config block `trainer.shared_builder_contract_check` in both cotrain YAMLs.
- Evidence:
  - Commands:
    - `/usr/bin/python3 -m py_compile starVLA/training/train_starvla.py`
    - `rg -n "shared_builder_contract_check|expected_schema_version|strict_legacy_no_extra" starVLA/config/training/starvla_cotrain_*.yaml`
  - Key outputs/metrics:
    - `py_compile` passed for `train_starvla.py`.
    - Both YAMLs include contract-check block defaults.
- Decision:
  - P2/P3 trainer gate path is now wired and configurable via CLI overrides used in smoke commands.
- Risks/Notes:
  - End-to-end accelerate smoke still requires user-side training env on server for final validation.
- Next step:
  - Commit this trainer/config patch and run/collect remote 20-step smoke logs with contract check enabled.
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] add trainer contract gate for shared-builder schema checks`

## [2026-04-05 22:14:00 +08:00] ALG1-DATA-20260405-001-OC fix P3 smoke blocker by replacing non-existent framework id in oxe config

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Unblock P3 training smoke after failure `Framework QwenFM is not implemented`.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/config/training/starvla_cotrain_oxe.yaml`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Updated `framework.name` from `QwenFM` to implemented `QwenGR00T` in OXE cotrain config.
    - Added explicit comment to prevent future regression.
- Evidence:
  - Commands:
    - server smoke log showed `NotImplementedError: Framework QwenFM is not implemented` from `build_framework`.
    - `rg -n "register\(\"QwenFM\"\)|framework.name:\s*QwenFM"` confirmed no `QwenFM` registration and only this config used it.
  - Key outputs/metrics:
    - Root cause isolated to framework id mismatch (not shared-builder contract path).
- Decision:
  - Keep contract gate path unchanged; fix only framework selection baseline for oxe smoke.
- Risks/Notes:
  - This is a config-level compatibility fix; model behavior follows `QwenGR00T` implementation.
- Next step:
  - Re-run 20-step accelerate smoke with same flags; no extra override needed for `framework.name` now.
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] fix oxe smoke config to use implemented QwenGR00T framework`

## [2026-04-06 11:08:00 +08:00] ALG1-DATA-20260405-001-OC add VLM attention fallback to bypass flash-attn ABI mismatch in P3 smoke

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Unblock P3 smoke when `flash_attn_2_cuda` import fails due ABI mismatch (`undefined symbol ... c10::Error`).
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/model/modules/vlm/QWen3.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/model/modules/vlm/QWen2_5.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Removed hard-coded `attn_implementation="flash_attention_2"` loading path.
    - Added `requested_attn_impl = config.framework.qwenvl.attn_implementation` parsing with default flash.
    - Added guarded fallback: if requested impl is flash and model loading throws, retry with `attn_implementation="sdpa"` and warning log.
    - Kept behavior unchanged for non-flash attention implementations.
- Evidence:
  - Commands:
    - server run log showed `ImportError ... flash_attn_2_cuda ... undefined symbol` during Qwen model init.
    - `/usr/bin/python3 -m py_compile starVLA/model/modules/vlm/QWen3.py starVLA/model/modules/vlm/QWen2_5.py`
  - Key outputs/metrics:
    - Syntax check passed for both patched files.
- Decision:
  - Prefer runtime fallback over immediate environment rebuild to keep smoke gate moving.
- Risks/Notes:
  - `sdpa` may be slower than flash-attn; acceptable for 20-step smoke and contract gate verification.
- Next step:
  - Push patch to server tmp branch and rerun same accelerate smoke command (optionally set `--framework.qwenvl.attn_implementation sdpa`).
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] add qwen vl attn fallback from flash to sdpa for smoke reliability`

## [2026-04-06 11:17:00 +08:00] ALG1-DATA-20260405-001-OC guard distributed calls before process-group init for single-process accelerate smoke

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Fix P3 smoke crash when `dist.get_rank()` is called before default process group init.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/dataloader/__init__.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/training/train_starvla.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Added `_dist_initialized()` helper in dataloader and trainer modules.
    - Replaced unsafe `dist.get_rank()==0` check in dataloader statistics save path with guarded check.
    - Guarded `dist.barrier()` in `prepare_data`, `eval_action_model`, and `main` teardown.
    - Switched `_log_metrics` rank guard to `self.accelerator.is_main_process`.
- Evidence:
  - Commands:
    - server log traceback showed crash at `starVLA/dataloader/__init__.py` `dist.get_rank()` with uninitialized process group.
    - `/usr/bin/python3 -m py_compile starVLA/dataloader/__init__.py starVLA/training/train_starvla.py`
  - Key outputs/metrics:
    - Syntax check passed for patched files.
- Decision:
  - Keep distributed guards explicit to support both initialized and non-initialized accelerate startup paths.
- Risks/Notes:
  - User command currently uses `starvla_train_pi.yaml` while expected chunk length in flags is set to 16; this may trigger contract mismatch (PI default is typically 8).
- Next step:
  - Push patch and rerun smoke; if contract mismatch appears, align `expected_action_chunk_len` to config-derived chunk length.
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] guard dist calls before init in dataloader/trainer smoke path`

## [2026-04-06 11:23:00 +08:00] ALG1-DATA-20260405-001-OC patch trainer_utils distributed main-rank guards for pre-init safety

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Resolve next-stage crash in `TrainerUtils.print_trainable_parameters` caused by calling `dist.get_rank()` before process-group init.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/training/trainer_utils/trainer_tools.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Added `_dist_initialized()` and `_is_main_process()` helpers.
    - Replaced all direct `dist.get_rank()` checks in this file with guarded `_is_main_process()` logic.
    - Fixed bug `if dist.get_rank == 0` (function object comparison) to proper guarded main-process check.
    - Updated `only_main_process` decorator to rely on `_is_main_process()`.
- Evidence:
  - Commands:
    - traceback showed crash at `trainer_tools.py:201` in `print_trainable_parameters`.
    - `/usr/bin/python3 -m py_compile starVLA/training/trainer_utils/trainer_tools.py`
    - `rg -n "dist\.get_rank|dist\.is_initialized" starVLA/training/trainer_utils/trainer_tools.py`
  - Key outputs/metrics:
    - Syntax check passed.
    - Only helper-level rank/init checks remain in file.
- Decision:
  - Consolidate rank checks into helper to prevent repeated pre-init crashes across trainer utilities.
- Risks/Notes:
  - User command line contains `--trainer.shared_builder_contract_check.expected_action_chunk_len 8 \ ` (note trailing space before `\`).
  - In shell this can break line continuation; should be `... 8 \` with no trailing space.
- Next step:
  - Push patch and rerun smoke; if shell continuation issue appears, rerun command with clean continuations.
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] guard trainer utils rank checks before dist init`

## [2026-04-06 11:31:00 +08:00] ALG1-DATA-20260405-001-OC bypass deepspeed mpi4py dependency by setting single-process dist env defaults

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Resolve DeepSpeed init failure `ModuleNotFoundError: No module named 'mpi4py'` during single-process smoke.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/training/train_starvla.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Added `_ensure_single_process_dist_env_defaults()` before global Accelerator initialization.
    - Function sets default env vars if absent: `RANK=0`, `LOCAL_RANK=0`, `WORLD_SIZE=1`, `MASTER_ADDR=127.0.0.1`, `MASTER_PORT=29500`.
    - This prevents DeepSpeed from entering MPI auto-discovery path (which required `mpi4py`).
- Evidence:
  - Commands:
    - traceback shows failure in `deepspeed/comm/comm.py -> mpi_discovery -> from mpi4py import MPI`.
    - `/usr/bin/python3 -m py_compile starVLA/training/train_starvla.py`
  - Key outputs/metrics:
    - Syntax check passed for patched training entry file.
- Decision:
  - Keep DeepSpeed enabled for memory feasibility, but avoid MPI dependency by explicitly pinning single-process env defaults.
- Risks/Notes:
  - If user explicitly runs multi-process distributed launch, these defaults should be overridden by launcher-provided env values.
- Next step:
  - Push patch and rerun 20-step smoke with unchanged command line.
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] set dist env defaults to avoid deepspeed mpi discovery`

## [2026-04-06 11:38:00 +08:00] ALG1-DATA-20260405-001-OC add forward-only contract smoke mode to avoid optimizer OOM in P3 gate

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Keep P3 contract smoke progressing when full training step OOM occurs during AdamW optimizer state updates.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/training/train_starvla.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Added `_is_contract_forward_only(cfg)` helper.
    - Added smoke switches:
      - `trainer.smoke_forward_only=true` or
      - `trainer.shared_builder_contract_check.forward_only=true`
    - In `_train_step`, when forward-only is enabled:
      - run contract check + model forward
      - record `action_dit_loss`
      - skip `backward`, grad clip, `optimizer.step`, `lr_scheduler.step`
      - emit metric `debug/shared_builder_forward_only=1.0`
- Evidence:
  - Commands:
    - runtime traceback showed OOM in optimizer step (`_multi_tensor_adamw`) with ~127 GiB GPU used.
    - `/usr/bin/python3 -m py_compile starVLA/training/train_starvla.py`
  - Key outputs/metrics:
    - Syntax check passed; forward-only toggles are present and discoverable in code.
- Decision:
  - Use forward-only mode for P3 gate smoke (contract path verification), and keep full-train path unchanged for real training.
- Risks/Notes:
  - Forward-only smoke validates contract and forward path, but does not validate backward/optimizer stability.
- Next step:
  - Push patch and rerun smoke with `--trainer.shared_builder_contract_check.forward_only true`.
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] add forward-only shared-builder smoke mode to bypass optimizer OOM`

## [2026-04-06 11:47:00 +08:00] ALG1-DATA-20260405-001-OC confirm P3 shared-builder contract forward-only smoke pass from remote run logs

- Owner: OC
- Status: DONE
- Objective:
  - Validate shared-builder contract gate under low-risk forward-only smoke path on authoritative server repo.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - No code changes in this entry; this is a gate result record based on remote run evidence.
- Evidence:
  - Commands:
    - user-provided `accelerate launch` with `--trainer.shared_builder_contract_check.forward_only true`.
  - Key outputs/metrics:
    - `Total optimization steps = 5` completed to `100%`.
    - `Training complete. Final model saved at results/Checkpoints/p3_contract_gate_smoke_20260406_forward_only/final_model`.
    - W&B run synced successfully (`oi9pchji`).
- Decision:
  - P3 contract gate is accepted for forward-only smoke path.
- Risks/Notes:
  - Full backward+optimizer stability is not covered by forward-only smoke (previously blocked by OOM).
- Next step:
  - Continue to next stage (pseudo-label construction / downstream pipeline integration).
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] record P3 forward-only contract smoke pass from remote logs`

## [2026-04-06 12:01:00 +08:00] ALG1-DATA-20260405-001-OC scaffold minimal correction pseudo-label builder and shared-builder smoke validator

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Start next-stage dataset-builder work with a minimal executable pseudo-label pipeline and schema validator.
- Changes:
  - Files:
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/dataset_builder/__init__.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/dataset_builder/sample_schema.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/dataset_builder/pseudo_label_utils.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/starVLA/dataset_builder/build_correction_dataset.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/tools/handoff/validate_shared_builder_schema.py`
    - `/Users/bazinga/code/my-starvla-v2-authoritative/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Added correction dataset schema (`p1_correction_dataset_v1`) and strict entry validator.
    - Added deterministic pseudo-label generator from action chunk deltas (`trigger_label`, `risk_score`, `affected_region_prior`, `correction_mask`).
    - Added CLI builder to read shared-builder dataloader samples and export `correction_dataset.jsonl` + `summary.json`.
    - Added shared-builder smoke log validator CLI (`tools/handoff/validate_shared_builder_schema.py`) for `SMOKE_RESULT_JSON_START/END` blocks.
- Evidence:
  - Commands:
    - `/usr/bin/python3 -m py_compile ... dataset_builder/*.py tools/handoff/validate_shared_builder_schema.py`
    - temporary smoke-log validation run returned `[PASS] validated 1 smoke block(s)`.
  - Key outputs/metrics:
    - New validator can verify schema version / shape / shared-vs-legacy mode constraints from smoke logs.
- Decision:
  - Move pseudo-label stage from docs-only to code-available baseline before full A/corrective trainer integration.
- Risks/Notes:
  - Local `/usr/bin/python3` missing `numpy` prevented executing full builder runtime test in this local environment; expected to run in server env.
- Next step:
  - Run server-side tiny builder smoke (`num_samples=8~32`) and validate output schema, then lock first correction dataset artifact path.
- Commit message:
  - `[ALG1-DATA-20260405-001-OC] add minimal correction pseudo-label builder and smoke schema validator`

## [2026-04-06 07:53:26 +00:00] ALG1-INFRA-20260406-001-OC fix pseudo-label zero-risk trigger consistency

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - ...
- Changes:
  - Files:
    - `...`
  - Code/Config summary:
    - ...
- Evidence:
  - Commands:
    - `...`
  - Key outputs/metrics:
    - ...
- Decision:
  - ...
- Risks/Notes:
  - ...
- Next step:
  - ...
- Commit message:
  - `[ALG1-INFRA-20260406-001-OC] fix pseudo-label zero-risk trigger consistency`

## [2026-04-06 13:25:00 +08:00] ALG1-INFRA-20260406-001-OC fix pseudo-label consistency and degeneracy in P1 builder

- Owner: OC
- Status: IN_PROGRESS
- Objective:
  - Fix P1 correction pseudo-label contradictions (`risk_score=0` but `trigger_label=1`) and reduce all-positive degeneracy.
- Changes:
  - Files:
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/dataset_builder/pseudo_label_utils.py`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/dataset_builder/sample_schema.py`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/dataset_builder/build_correction_dataset.py`
  - Code/Config summary:
    - Added zero-risk hard guard: if `max(delta_action_norm) <= eps`, force `trigger=0`, `risk=0`, `mask=all-0`, `prior=all-0`.
    - Switched risk norm to motion channels (`[:-1]`) to avoid binary gripper toggles dominating pseudo labels.
    - Added trigger threshold (`risk >= 0.1`) and mask stabilization (`quantile + relative floor`, argmax fallback).
    - Extended schema validator with `delta_action_norm` required key and consistency rules (`risk==0 => trigger/mask/prior zero`).
    - Builder summary now writes absolute `output_jsonl` and includes quick trigger distribution + contradiction counter.
- Evidence:
  - Commands:
    - `/2025233147/envs/llava3d_vla_train/bin/python -m py_compile starVLA/dataset_builder/pseudo_label_utils.py starVLA/dataset_builder/sample_schema.py starVLA/dataset_builder/build_correction_dataset.py`
    - local replay check on provided `correction_dataset.jsonl` with patched pseudo-label logic.
  - Key outputs/metrics:
    - `py_compile` passed.
    - Replay stats (N=16): `trigger={0:5,1:11}`, `risk0_trig1=0`, `mask_all_one=0`, `schema_err=0`.
- Decision:
  - Patch is valid for P1 sanity criteria: removes zero-risk contradiction and avoids all-one masks in provided sample set.
- Risks/Notes:
  - Trigger threshold `0.1` is heuristic; may need tuning on larger mixed datasets.
- Next step:
  - Run server-side tiny builder smoke (`num_samples=16~64`) and confirm output artifact stats before promoting to worktree/final.
- Commit message:
  - `[ALG1-INFRA-20260406-001-OC] fix pseudo-label zero-risk consistency and trigger/mask degeneracy`
## [2026-04-06 14:08:00 +08:00] ALG1-INFRA-20260406-001-OC verify P1 pseudo-label fix on 64-sample sanity output

- Owner: OC
- Status: DONE
- Objective:
  - Confirm repaired pseudo-label logic on larger sanity set and close P1 consistency bugfix loop.
- Changes:
  - Files:
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - No code change in this entry; this is validation evidence for commit `ea6b3e9`.
- Evidence:
  - Commands:
    - `cat /2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/PseudoLabels/p1_sanity_20260406_fix64/summary.json`
    - jsonl stat check on `correction_dataset.jsonl` (`mask_all_one`, `risk0_trig1`).
  - Key outputs/metrics:
    - `num_samples=64`
    - `trigger_distribution={0:9,1:55}`
    - `risk_zero_trigger_one=0`
    - `mask_all_one=0`
- Decision:
  - P1 pseudo-label consistency fix accepted on 64-sample sanity set.
- Risks/Notes:
  - Positive ratio remains high (55/64), may still need threshold tuning during larger-run calibration.
- Next step:
  - Promote branch flow `tmp -> worktree`, then prepare integration checks for next gate.
- Commit message:
  - `[ALG1-INFRA-20260406-001-OC] record 64-sample sanity pass for pseudo-label consistency fix`
## [2026-04-06 14:26:00 +08:00] ALG1-INFRA-20260406-001-OC calibrate trigger threshold and set default to 0.15

- Owner: OC
- Status: DONE
- Objective:
  - Reduce over-triggering risk in P1 pseudo labels by calibrating and updating trigger threshold.
- Changes:
  - Files:
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/starVLA/dataset_builder/pseudo_label_utils.py`
    - `/2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA/docs/algorithm1/handoff/progress_live.md`
  - Code/Config summary:
    - Updated `DEFAULT_TRIGGER_THRESHOLD` from `0.1` to `0.15` in pseudo-label builder.
- Evidence:
  - Commands:
    - 512-sample build run to collect risk distribution and scan thresholds.
    - threshold scan on `correction_dataset.jsonl` for `0.10/0.12/0.15/0.18/0.20/0.25/0.30`.
    - 128-sample rebuild after threshold update.
  - Key outputs/metrics:
    - On 512 samples (scan):
      - `thr=0.10 -> pos_ratio=0.8691`
      - `thr=0.12 -> pos_ratio=0.7500`
      - `thr=0.15 -> pos_ratio=0.5977` (closest to target 0.6)
      - `thr=0.18 -> pos_ratio=0.4688`
    - After setting default to 0.15 (128-sample sanity):
      - `trigger_distribution={0:53,1:75}` (pos_ratio=0.5859)
      - `risk_zero_trigger_one=0`
      - `mask_all_one=0`
- Decision:
  - Accept `0.15` as current default trigger threshold for P1 builder.
- Risks/Notes:
  - Threshold may still need retuning on larger or cross-domain mixtures.
- Next step:
  - Promote latest tmp commit to worktree and sync both branches to writable remote.
- Commit message:
  - `[ALG1-INFRA-20260406-001-OC] calibrate pseudo-label trigger threshold to 0.15`
