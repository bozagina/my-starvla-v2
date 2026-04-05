# StarVLA Retrofit: Prompt and Branch Strategy

- EXP_ID: `ALG1-INFRA-20260402-001-OC`
- Last major update: `ALG1-INFRA-20260403-001-OC`
- Scope: local development + remote training validation
- Active VLM scope: `Qwen2.5VL/Qwen3VL` only for this repo stream (`MapAnything/LLaVA3D` decoupled to external project)
- Source task book:
  - `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/star_vla改造与统一伪标签生成任务书.md`
- Companion compact startup:
  - `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/new_chat_bootstrap_compact.md`
- Companion operational skill:
  - `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/skills/starvla-retrofit-ops/SKILL.md`
- Companion deadlock guard:
  - `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/acceptance_deadlock_guard.md`
- Companion thread index:
  - `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`

---

## 1. Task-Book Understanding (Actionable)

This retrofit is not a rewrite project. The target is a unified StarVLA R&D base with four decoupled layers:

1. `base_chunk_policy` training/output
2. unified `dataset_builder` for pseudo labels
3. `A_module` training
4. `corrective_policy` training

Hard constraints from the task book:

1. Keep changes minimally invasive to existing StarVLA training flow.
2. A and corrective policy must share one sample builder (single source of truth).
3. Train in phases first; optional joint finetune later.
4. Async runtime compatibility should be interface-first, not tightly coupled now.

---

## 2. Branch Compatibility Analysis

Current branch graph snapshot:

1. `main` is clean baseline (branch head not behind others).
2. `codex/v4-1-delta-action-head` is `65` commits ahead of `main`, heavily coupled to Path-A/mask experiments.
3. `dev-SA-GRFlow-v1` is `42` commits ahead of `main`, also includes Path-A-era changes.
4. Working tree currently has uncommitted local edits in eval files.

Recommendation:

1. **Primary development base: `main`**
2. Start a new branch for this retrofit stream:
   - `codex/starvla-retrofit-unified-pseudolabel-20260402`
3. Cherry-pick only strictly needed utility commits later (if any), instead of inheriting all Path-A history.

Why:

1. Task-book goal is framework retrofit, not continuation of Path-A-specific tuning.
2. Starting from `main` avoids hidden coupling and reduces rollback risk.
3. New branch from `main` keeps paper-1 mainline decision path auditable.

---

## 3. Safe Branch Creation (Given Dirty Worktree)

Because current worktree has local modified files, prefer a separate worktree to avoid accidental carry-over.

```bash
cd /Users/bazinga/code/my-starvla-v2
git worktree add ../my-starvla-v2-retrofit \
  -b codex/starvla-retrofit-unified-pseudolabel-20260402 \
  main
```

Then develop in:

```bash
cd /Users/bazinga/code/my-starvla-v2-retrofit
```

Then initialize the 3-level branch model in that worktree:

```bash
git switch -c codex/worktree-starvla-v2-mainline
git switch -c codex/tmp-retrofit-<date>-<topic>
```

Branch responsibilities:

1. `codex/tmp-*`:
   - temporary patching branch for one small task
   - can be rebased/rewritten locally
2. `codex/worktree-*`:
   - stable integration branch of this worktree
   - only receives validated merges from `tmp` branches
3. final target branch (for cross-worktree or long-running integration):
   - receives milestone-level merges from `codex/worktree-*`

Promotion flow:

```bash
# after local validation on tmp branch
git switch codex/worktree-starvla-v2-mainline
git merge --no-ff codex/tmp-retrofit-<date>-<topic>

# milestone handoff to final branch
git switch codex/<final-target-branch>
git merge --no-ff codex/worktree-starvla-v2-mainline
```

Optional alternative (only if current edits can be stashed/committed safely):

```bash
git switch main
git switch -c codex/worktree-starvla-v2-mainline
git switch -c codex/tmp-retrofit-<date>-<topic>
```

---

## 4. Prompt Retrofit (Current v2 Contract)

Current mask-stream prompt is not enough for retrofit execution. Use retrofit-specific compact startup + operating contract.

### 4.1 Mandatory startup block (v2)

1. Read compact context first:
   - `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/context_pack_compact.md`
   - `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
2. Read deep task-book only if compact context is insufficient.
3. Check branch/dirty status and confirm three-level topology:
   - `codex/tmp-* -> codex/worktree-* -> codex/<final-target>`
4. Confirm current phase:
   - `P0 audit`
   - `P1 builder`
   - `P2 A trainer`
   - `P3 corrective trainer`
5. Create EXP_ID before first patch.
6. Optional helper commands:
   - `/Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh prompt-retrofit`
   - `bash /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh`

### 4.2 Mandatory delivery format per round

Always report:

1. what changed
2. evidence collected
3. conclusion supported
4. next recommended action
5. commit message in `[EXP_ID] one-line intent`

### 4.3 Do-not-drift rules

1. Do not jump to async runtime implementation in this phase.
2. Do not start joint end-to-end training before phased pipeline is stable.
3. Do not let A and corrective policy use different pseudo-label generators.
4. Do not make branch-wide refactors before P0 audit checklist is closed.
5. Do not start Path-A related expansion before `P0` gate closure.
6. Do not re-introduce `MapAnything/LLaVA3D` as required dependency in prompts/checklists/default scripts in this stream.

---

## 5. Suggested New Prompt Template (Copy Block)

```text
You are continuing StarVLA retrofit in /Users/bazinga/code/my-starvla-v2.

Mission:
- Keep minimal-invasive changes on StarVLA.
- Build one shared dataset builder for A module and corrective policy.
- Execute phase-by-phase:
  P0 audit -> P1 dataset builder -> P2 A trainer -> P3 corrective trainer.

Startup (mandatory):
1) Read:
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/context_pack_compact.md
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md
   - /Users/bazinga/code/my-starvla-v2/docs/algorithm1/handoff/progress_live.md (tail only)
2) Load deep task-book only if current gate needs more detail:
   - /Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/handoff/star_vla改造与统一伪标签生成任务书.md
3) Check current branch status and three-level topology:
   - codex/tmp-* -> codex/worktree-* -> codex/<final-target>
4) Create EXP_ID before first code patch:
   - /Users/bazinga/code/my-starvla-v2/tools/handoff/bootstrap_session.sh start --module INFRA --owner OC --title "<one line task title>"
5) Keep current gate explicit and do not skip gate without file-level evidence.

Hard constraints:
- Local dev / remote train boundary is strict.
- Final remote-effect conclusions rely on user-provided remote logs/config/metrics.
- One sample builder is single source of truth.
- No Path-A expansion before P0 closure.
- VLM track in this repo stream is Qwen-only (`Qwen2.5VL/Qwen3VL`), and MapAnything/LLaVA3D references are historical context only.
- Every major modification must append progress log.

Output each round:
1) What changed
2) Evidence
3) Conclusion
4) Next step
5) [EXP_ID] commit message
```

---

## 6. Skill + Token-Efficiency Addendum

Use project skill playbook for repeatability:

- `/Users/bazinga/code/my-starvla-v2/docs/starvla_retrofit/skills/starvla-retrofit-ops/SKILL.md`

Token-efficiency principles in this stream:

1. Tiered context loading (`compact first`, `deep on demand`).
2. Keep round status within 8 bullets unless user asks for detail.
3. Prefer file-level evidence and commands over long narrative.
4. Use fixed round schema: changes / evidence / conclusion / next step / commit message.

---

## 7. First Practical Execution Plan

1. `P0-Audit`: confirm chunk output capability and episode replay capability.
2. `P0-Audit`: confirm dataloader window support and trainer extensibility.
3. `P1-Builder`: implement unified sample schema + pseudo-label generator.
4. `P1-Builder`: produce a tiny sanity dataset and schema validator.
5. `P2/P3`: scaffold trainers with fixed input-output protocol and minimal losses.

Exit criteria for moving out of P0:

1. audit checklist completed with file-level evidence.
2. branch and module map frozen.
3. no blocker on trainer insertion path.
