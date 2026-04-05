# Context Loading Order (Token-Efficient)

## Goal

Load the minimum context required to make a safe decision, then expand only when blocked.

## Tier 0 (Always)

1. `docs/starvla_retrofit/handoff/context_pack_compact.md`
2. `docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
3. `docs/algorithm1/handoff/progress_live.md` (tail section only)

## Tier 1 (Phase-Scoped)

Load only one branch based on current phase:

- `P0`: trainer entry + dataloader + framework action head
- `P1`: dataset builder files + schema files
- `P2`: A module trainer + module I/O protocol
- `P3`: corrective trainer + integration I/O protocol

## Tier 2 (Exception-Only)

Load deep historical docs only when Tier 0/1 cannot resolve ambiguity:

- full task-book sections
- large history docs
- legacy path-A docs

## Stop Rules

1. If decision can be made with Tier 0/1, do not load Tier 2.
2. If loaded docs exceed what current task needs, summarize and close them.
3. Prefer file-level evidence snippets over broad narrative summary.
