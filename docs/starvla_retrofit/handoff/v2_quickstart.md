# my-starvla-v2 Quickstart (Worktree + Temp Branch Flow)

This repository uses a 3-level branch promotion model:

1. `codex/tmp-*`:
   - temporary patch branch
   - short-lived, task-focused
2. `codex/worktree-*`:
   - stable integration branch for this worktree
   - merges validated tmp branches only
3. final target branch:
   - milestone-level integration branch
   - merges from worktree branch

Active model scope for this repo stream:

1. `Qwen2.5VL` / `Qwen3VL` only.
2. `MapAnything/LLaVA3D/3D-VLM` belongs to another decoupled project and should not be added as mandatory dependency here.

---

## 1) Daily Start

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export STARVLA_EXPECTED_REPO_ROOT="<absolute-target-repo-root>"
export STARVLA_EXPECTED_VLM_SCOPE="qwen_only"
bash "$REPO_ROOT/tools/handoff/ensure_repo_context.sh" --expect-root "$STARVLA_EXPECTED_REPO_ROOT" --expect-vlm-scope "$STARVLA_EXPECTED_VLM_SCOPE" --require-expected-root
git fetch origin
git switch codex/worktree-starvla-v2-mainline
git merge --ff-only origin/main
git switch -c codex/tmp-<YYYYMMDD>-<task>
```

---

## 2) Task Development

1. Develop and validate on `codex/tmp-*`.
2. Keep patches small and evidence-first.
3. Append major updates to progress log.

---

## 3) Promote to Worktree Branch

```bash
git switch codex/worktree-starvla-v2-mainline
git merge --no-ff codex/tmp-<YYYYMMDD>-<task>
```

---

## 4) Promote to Final Branch

```bash
git switch codex/<final-target-branch>
git merge --no-ff codex/worktree-starvla-v2-mainline
```

---

## 5) Required Docs for This Stream

1. `<REPO_ROOT>/docs/starvla_retrofit/handoff/retrofit_prompt_and_branch_strategy.md`
2. `<REPO_ROOT>/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md`
3. `<REPO_ROOT>/docs/starvla_retrofit/handoff/star_vla改造与统一伪标签生成任务书.md`
4. `<REPO_ROOT>/docs/algorithm1/handoff/progress_live.md`
5. `<REPO_ROOT>/docs/starvla_retrofit/handoff/development_run_checklist.md`
6. `<REPO_ROOT>/docs/starvla_retrofit/handoff/thread_prompts_and_checklists_index.md`
7. `<REPO_ROOT>/docs/starvla_retrofit/handoff/ALG1-INFRA-20260403-006-OC__qwen_only_prompt_system_cleanup_audit.md`
