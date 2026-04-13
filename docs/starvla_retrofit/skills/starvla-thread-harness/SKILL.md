---
name: starvla-thread-harness
description: Run StarVLA Harness v2 threads with explicit module-role identity, manifest-first execution, frozen-anchor discipline, evidence-only review, and upstream/downstream blocking rules for A-module and corrective policy parallel development.
---

# StarVLA Thread Harness

## Overview

Use this skill whenever the task belongs to one of the v2 threads:

1. `A-RES`
2. `A-BUILD`
3. `A-REVIEW`
4. `CP-RES`
5. `CP-BUILD`
6. `CP-REVIEW`

## Required Startup

1. Lock repo identity first.
2. Run readiness.
3. State `module`, `role`, `thread_id`, and `current_gate`.
4. Load the active manifest before any code edits.

## Required Context

Always load:

1. `docs/starvla_retrofit/handoff/thread_operator_quick_reference.md` (operator entry point)
2. `docs/starvla_retrofit/handoff/harness_v2_overview.md`
3. `docs/starvla_retrofit/handoff/harness_v2_thread_matrix.md`
4. `docs/starvla_retrofit/handoff/new_chat_bootstrap_thread_v2.md`
5. active manifest under `docs/algorithm1/handoff/manifests/`
6. current tail of `docs/algorithm1/handoff/progress_live.md`

Load role-specific docs only after the above.

## Non-Negotiable Rules

1. Do not change thread identity mid-round.
2. Do not redefine frozen baselines inside `BUILD`.
3. Do not let `REVIEW` approve from intent alone.
4. If upstream is not ready, mark `BLOCKED_WAIT_UPSTREAM`.
5. If remote artifacts are missing, mark `BLOCKED_WAIT_REMOTE`.

## Output Shape

Every substantial round should report:

1. What changed
2. What evidence was collected
3. What conclusion is supported
4. What is still blocked or unknown
5. What exact downstream handoff exists
