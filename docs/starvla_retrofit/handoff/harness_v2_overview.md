# StarVLA Parallel Harness v2

## Purpose

This document upgrades the current single-stream retrofit workflow into a module-centric parallel harness suitable for:

1. `A-module`
2. `corrective policy`

Each module is split into three long-lived thread roles:

1. `RES`: research
2. `BUILD`: implementation
3. `REVIEW`: independent review

The result is a six-thread matrix:

1. `A-RES`
2. `A-BUILD`
3. `A-REVIEW`
4. `CP-RES`
5. `CP-BUILD`
6. `CP-REVIEW`

## Why v2 Exists

The original retrofit harness is phase-centered:

- `P0 audit`
- `P1 builder`
- `P2 A trainer`
- `P3 corrective trainer`

That works for one main stream, but it drifts once:

1. `A-module` and `corrective policy` evolve in parallel
2. research and build threads start making cross-module decisions
3. review threads are asked to arbitrate both engineering and semantic claims

Harness v2 fixes that by making every thread explicit on three axes:

1. `module`
2. `role`
3. `gate`

## Core Rules

1. Every thread must have a manifest.
2. Every thread must declare its baseline anchor.
3. Every thread must declare the contract version it consumes and emits.
4. `BUILD` cannot redefine frozen baselines.
5. `REVIEW` cannot treat "feature exists" as "acceptance passed".
6. `CP-BUILD` cannot start until `A-REVIEW` explicitly says A output is usable as upstream input.

## What Stays The Same

Harness v2 does not replace existing repo safety mechanisms. It still depends on:

1. `tools/handoff/ensure_repo_context.sh`
2. `tools/handoff/pre_dev_readiness.sh`
3. `tmp -> worktree -> final` promotion
4. evidence-backed reporting
5. `BLOCKED_WAIT_REMOTE` deadlock discipline

## New Objects Introduced By v2

1. `thread manifest`
2. module-specific progress ledgers
3. a common six-thread harness skill
4. six role-specific skills
5. a dedicated thread bootstrap helper

## Thread Bootstrap Flow

1. Run repo guard and readiness.
2. Select `module`, `role`, and `current_gate`.
3. Create a thread session manifest.
4. Append one entry to the global progress log.
5. Append one entry to the module-specific progress log.
6. Execute only within that thread's allowed scope.

## Files Introduced By v2

Primary docs:

1. `docs/starvla_retrofit/handoff/harness_v2_overview.md`
2. `docs/starvla_retrofit/handoff/harness_v2_thread_matrix.md`
3. `docs/starvla_retrofit/handoff/a_module_task_flow_v2.md`
4. `docs/starvla_retrofit/handoff/new_chat_bootstrap_thread_v2.md`
5. `docs/starvla_retrofit/handoff/thread_operator_quick_reference.md`

Progress ledgers:

1. `docs/algorithm1/handoff/progress_a_module.md`
2. `docs/algorithm1/handoff/progress_corrective_policy.md`

Manifests:

1. `docs/algorithm1/handoff/manifests/a_module_current_round.yaml`
2. `docs/algorithm1/handoff/manifests/cp_current_round.yaml`

Bootstrap and validation tools:

1. `tools/handoff/bootstrap_session.sh` (subcommand `prompt-thread-v2`)
2. `tools/handoff/pre_dev_readiness.sh` (v2 checks via `STARVLA_THREAD_ID`)
3. `tools/handoff/validate_thread_v2.py`

Skills:

1. `docs/starvla_retrofit/skills/starvla-thread-harness/SKILL.md`
2. `docs/starvla_retrofit/skills/a-module-research/SKILL.md`
3. `docs/starvla_retrofit/skills/a-module-build/SKILL.md`
4. `docs/starvla_retrofit/skills/a-module-review/SKILL.md`
5. `docs/starvla_retrofit/skills/corrective-policy-research/SKILL.md`
6. `docs/starvla_retrofit/skills/corrective-policy-build/SKILL.md`
7. `docs/starvla_retrofit/skills/corrective-policy-review/SKILL.md`

## Adoption Guidance

1. Use v1 harness for historical retrofit-only work.
2. Use v2 harness for any work that touches `A-module`, `corrective policy`, or their handoff boundary.
3. Treat old `T1/T2/T3/T4` docs as internal implementation splits, not top-level thread identity.
