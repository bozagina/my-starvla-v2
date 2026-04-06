---
name: starvla-retrofit-ops
description: Execute StarVLA retrofit tasks with strict phase gates from P0 to P3, three-level branch promotion from tmp to worktree to final, evidence-backed reporting, and token-efficient context loading. Use for audit, pseudo-label builder planning, trainer insertion planning, and handoff document upkeep in my-starvla-v2.
---

# StarVLA Retrofit Ops

## Overview

Use this skill to run StarVLA retrofit work with minimal context cost, reproducible evidence, and consistent handoff artifacts.

## Workflow

1. Run preflight.
- Lock target repo identity first:
  - export `REPO_ROOT="$(git rev-parse --show-toplevel)"`
  - export `STARVLA_EXPECTED_REPO_ROOT="<absolute-target-repo-root>"`
  - export `STARVLA_EXPECTED_VLM_SCOPE="qwen_only"`
  - run `tools/handoff/ensure_repo_context.sh --expect-root ... --expect-vlm-scope qwen_only --require-expected-root`
- Check branch and worktree cleanliness.
- Confirm `tmp -> worktree -> final` topology exists.
- Create EXP_ID before first patch.

2. Load context in two tiers.
- Tier 0 (always):
  - `docs/starvla_retrofit/handoff/context_pack_compact.md`
  - `docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
  - `docs/algorithm1/handoff/progress_live.md` (tail only)
- Tier 1 (only when needed):
  - task-book deep sections
  - module source files related to current gate

3. Execute by phase gate.
- `P0`: audit only (chunk/replay/window/trainer insertion).
- `P1`: shared dataset builder and schema only.
- `P2`: A module trainer.
- `P3`: corrective policy trainer.
- Do not skip gate without file-level evidence.

4. Produce round output in fixed structure.
- What changed
- Evidence collected
- Conclusion supported
- Next action
- Commit message: `[EXP_ID] one-line intent`

5. Keep token usage bounded.
- Prefer compact bullet summaries over long narrative.
- Reuse templates in `references/`.
- Load only phase-relevant source files.

## Commands

Create EXP_ID:

```bash
"$REPO_ROOT/tools/handoff/bootstrap_session.sh" start \
  --module INFRA \
  --owner OC \
  --title "<one line title>"
```

Run preflight helper:

```bash
bash "$REPO_ROOT/docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh"
```

## References

- Context loading policy:
  - `references/context_loading_order.md`
- Round report template:
  - `references/round_report_template.md`
- P1 file plan template:
  - `references/p1_plan_by_file_template.md`
