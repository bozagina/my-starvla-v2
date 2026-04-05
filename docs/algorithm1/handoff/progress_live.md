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
