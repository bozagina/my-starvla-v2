# StarVLA Retrofit Context Pack (Compact)

Purpose: provide the minimum stable context for new sessions with lower token cost.

## Current Objective

Build a minimally invasive retrofit on `my-starvla-v2` that supports:
1. base chunk policy
2. shared pseudo-label builder
3. A module training
4. corrective policy training

## Hard Constraints

1. Follow phase gates in order: `P0 -> P1 -> P2 -> P3`.
2. Do not skip gate without file-level evidence.
3. Enforce one shared builder as single source of truth for A + corrective.
4. Keep local-dev vs remote-train boundary explicit.
5. Do not start async runtime deep coupling in this stream.
6. Do not start path-A expansion before `P0` is closed.
7. Use branch promotion discipline: `codex/tmp-* -> codex/worktree-* -> codex/<final-target>`.
8. Every round must include: changes, evidence, conclusion, next action, commit message.
9. Active VLM scope is `Qwen2.5VL/Qwen3VL` only; `MapAnything/LLaVA3D` content is historical/out-of-scope in this repo stream.

## Startup Sequence

1. Check git branch + dirty state.
2. Create EXP_ID via `tools/handoff/bootstrap_session.sh start ...`.
3. Load only phase-relevant source files.
4. Append major updates to `docs/algorithm1/handoff/progress_live.md`.

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
