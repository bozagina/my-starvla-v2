# ALG1-INFRA-20260403-001-OC Skill + Prompt System Upgrade

## 1) Objective

Find the best available skill strategy for StarVLA retrofit, then upgrade prompt/docs so future development is:
- easier for human review
- easier for model execution
- lower token cost per round

## 2) Search Scope and Evidence

### Primary sources

1. OpenAI curated skills list (official):
- https://github.com/openai/skills/tree/main/skills/.curated

2. OpenAI Codex docs (official):
- AGENTS.md guide: https://developers.openai.com/codex/guides/agents-md
- Skills guide: https://developers.openai.com/codex/skills

3. OpenAI Codex repo docs (official pointers):
- https://github.com/openai/codex/blob/main/docs/agents_md.md
- https://github.com/openai/codex/blob/main/docs/skills.md

4. External high-signal community reference:
- https://github.com/feiskyer/codex-settings

### Local command evidence

- `list-skills.py --format json` returned current curated set.
- `list-skills.py --path skills/.experimental` returned path-not-found (current upstream only has `.curated` and `.system`).
- GitHub API check confirmed `openai/skills` currently exposes `skills/.curated` and `skills/.system`.

## 3) Candidate Skill Evaluation

Scoring scale: 1 (low) to 5 (high).

| Candidate | Retrofit relevance | Prompt/doc relevance | Token-efficiency support | Overall |
|---|---:|---:|---:|---:|
| `openai-docs` (curated/system) | 3 | 4 | 3 | 3.3 |
| `doc` (curated) | 2 | 3 | 2 | 2.3 |
| `gh-fix-ci` (curated) | 2 | 2 | 3 | 2.3 |
| `skill-creator` (system) | 5 | 5 | 5 | 5.0 |
| Project-specific custom skill (based on above) | 5 | 5 | 5 | 5.0 |

Conclusion:
- No single curated skill directly matches StarVLA retrofit gate workflow.
- Best strategy is: use official skill principles + build project-specific custom skill.

## 4) Implemented Upgrade

### 4.1 New project skill

Added:
- `docs/starvla_retrofit/skills/starvla-retrofit-ops/SKILL.md`
- `docs/starvla_retrofit/skills/starvla-retrofit-ops/agents/openai.yaml`
- `docs/starvla_retrofit/skills/starvla-retrofit-ops/references/context_loading_order.md`
- `docs/starvla_retrofit/skills/starvla-retrofit-ops/references/round_report_template.md`
- `docs/starvla_retrofit/skills/starvla-retrofit-ops/references/p1_plan_by_file_template.md`
- `docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh`

Validation:
- `quick_validate.py` passed for this skill.

### 4.2 Prompt and context compaction

Added:
- `docs/starvla_retrofit/handoff/context_pack_compact.md`
- `docs/starvla_retrofit/handoff/new_chat_bootstrap_compact.md`

Updated:
- `tools/handoff/bootstrap_session.sh`
  - new command: `prompt-retrofit`
  - prints compact retrofit bootstrap prompt

## 5) Design Choices for Token Savings

1. Tiered context loading (Tier0/Tier1/Tier2) instead of full-doc preload.
2. Fixed round report schema to reduce narrative verbosity.
3. Phase-scoped file reading (only gate-relevant files).
4. Compact bootstrap prompt for retrofit stream.
5. Preflight script for repeated checks instead of long manual instructions.

## 6) How to Use Now

1. Print compact retrofit startup prompt:
```bash
/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh prompt-retrofit
```

2. Run preflight checks:
```bash
bash /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh
```

3. Use skill docs as operational playbook:
- `docs/starvla_retrofit/skills/starvla-retrofit-ops/SKILL.md`

## 7) Risks and Follow-up

1. The custom skill currently lives in repo docs (versioned), not auto-installed under `~/.codex/skills`.
2. If automatic invocation is needed, add a follow-up install/symlink step.
3. P0/P1/P2/P3 technical implementation work remains phase-gated and separate.
