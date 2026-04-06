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
