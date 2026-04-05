# StarVLA Retrofit Context Pack (Compact)

Purpose: provide the minimum stable context for new sessions with lower token cost.

## Current Objective

Build a minimally invasive retrofit on `my-starvla-v2` that supports:
1. base chunk policy
2. shared pseudo-label builder
3. A module training
4. corrective policy training

## Hard Constraints

1. Session must lock explicit target repo first (`STARVLA_EXPECTED_REPO_ROOT`) and pass `tools/handoff/ensure_repo_context.sh`.
2. If repo identity check fails, stop immediately; do not continue audit/coding in that workspace.
3. Follow phase gates in order: `P0 -> P1 -> P2 -> P3`.
4. Do not skip gate without file-level evidence.
5. Enforce one shared builder as single source of truth for A + corrective.
6. Keep local-dev vs remote-train boundary explicit.
7. Do not start async runtime deep coupling in this stream.
8. Do not start path-A expansion before `P0` is closed.
9. Use branch promotion discipline: `codex/tmp-* -> codex/worktree-* -> codex/<final-target>`.
10. Every round must include: changes, evidence, conclusion, next action, commit message.
11. Active VLM scope is `Qwen2.5VL/Qwen3VL` only; `MapAnything/LLaVA3D` content is historical/out-of-scope in this repo stream.

## Startup Sequence

1. Set repo lock variables and run guard:
   - `export REPO_ROOT="$(git rev-parse --show-toplevel)"`
   - `export STARVLA_EXPECTED_REPO_ROOT="<absolute-target-repo-root>"`
   - `export STARVLA_EXPECTED_VLM_SCOPE="qwen_only"`
   - `bash "$REPO_ROOT/tools/handoff/ensure_repo_context.sh" --expect-root "$STARVLA_EXPECTED_REPO_ROOT" --expect-vlm-scope "$STARVLA_EXPECTED_VLM_SCOPE" --require-expected-root`
2. Check git branch + dirty state.
3. Create EXP_ID via `tools/handoff/bootstrap_session.sh start ...`.
4. Load only phase-relevant source files.
5. Append major updates to `docs/algorithm1/handoff/progress_live.md`.

## Deep Docs (Load Only If Needed)

- task-book:
  - `docs/starvla_retrofit/handoff/star_vla改造与统一伪标签生成任务书.md`
- contract:
  - `docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
- deadlock guard:
  - `docs/starvla_retrofit/handoff/acceptance_deadlock_guard.md`
- dev checklist:
  - `docs/starvla_retrofit/handoff/development_run_checklist.md`
- thread index:
  - `docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
- latest baseline report:
  - `docs/starvla_retrofit/handoff/ALG1-INFRA-20260403-006-OC__qwen_only_prompt_system_cleanup_audit.md`
- branch strategy:
  - `docs/starvla_retrofit/handoff/retrofit_prompt_and_branch_strategy.md`
