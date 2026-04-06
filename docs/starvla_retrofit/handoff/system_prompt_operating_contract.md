# System Prompt Operating Contract (StarVLA Retrofit Stream)

Use this contract for sessions focused on StarVLA retrofit and unified pseudo-label generation.

---

## 1) Mission

Build a minimally invasive StarVLA retrofit that supports:

1. base chunk policy
2. shared pseudo-label dataset builder
3. A module training
4. corrective policy training

Current VLM scope in this repository:

1. `Qwen2.5VL` / `Qwen3VL` only.
2. `MapAnything/LLaVA3D/3D-VLM` stream is decoupled to another project and is out of scope for this retrofit stream.

---

## 2) Mandatory Startup

Before coding:

1. Resolve and lock repo context first (hard gate):
   - `export REPO_ROOT="$(git rev-parse --show-toplevel)"`
   - `export STARVLA_EXPECTED_REPO_ROOT="<absolute-target-repo-root>"`
   - `export STARVLA_EXPECTED_VLM_SCOPE="qwen_only"`
   - `bash "$REPO_ROOT/tools/handoff/ensure_repo_context.sh" --expect-root "$STARVLA_EXPECTED_REPO_ROOT" --expect-vlm-scope "$STARVLA_EXPECTED_VLM_SCOPE" --require-expected-root`
2. Read compact context first:
   - `<REPO_ROOT>/docs/starvla_retrofit/handoff/context_pack_compact.md`
3. Read deep task-book only if compact context is insufficient:
   - `<REPO_ROOT>/docs/starvla_retrofit/handoff/star_vla改造与统一伪标签生成任务书.md`
4. Check branch/dirty status.
5. Confirm 3-level branch topology is available for current worktree:
   - `codex/tmp-*` (task patch branch)
   - `codex/worktree-*` (stable worktree integration)
   - `codex/<final-target>` (milestone integration target)
6. State current phase (`P0/P1/P2/P3`) and one-sentence goal.
7. Create EXP_ID before first patch:
   - `<REPO_ROOT>/tools/handoff/bootstrap_session.sh start --module INFRA --owner OC --title "<task title>"`

---

## 3) Phase Gates

1. `P0 Audit`:
   - chunk output support
   - episode replay support
   - dataloader window support
   - trainer insertion feasibility
2. `P1 Builder`:
   - one shared pseudo-label generator
   - shared schema for A and corrective policy
3. `P2 A Trainer`
4. `P3 Corrective Trainer`

Do not skip gates without evidence.

---

## 4) Non-Negotiable Rules

1. Minimal invasive changes first.
2. One sample builder is the only source of truth.
3. No async runtime deep coupling in this phase.
4. No early end-to-end joint training before phased pipeline is stable.
5. Local/remote boundary must be explicit.
6. All coding starts on `codex/tmp-*`; only validated changes are promoted to `codex/worktree-*`, then milestone-merged into final target branch.
7. Any new prompt/checklist/whitelist in this repo must align to `Qwen2.5VL/Qwen3VL` and must not require `MapAnything/LLaVA3D` files.

---

## 5) Output Standard (Every Round)

Always provide:

1. what changed
2. evidence collected
3. conclusion supported
4. next action
5. suggested commit message:
   - `[<EXP_ID>] <one-line intent>`

Append major modifications to:

- `<REPO_ROOT>/docs/algorithm1/handoff/progress_live.md`

---

## 6) Token-Efficiency Rules

1. Prefer compact context pack + phase-scoped source loading over full-doc preload.
2. Keep status summaries within 8 bullets unless user asks for deep detail.
3. Use file-level evidence and command traces instead of long prose.
4. Reuse structured templates (what changed / evidence / conclusion / next action / commit message).
5. If more context is needed, load incrementally and record why.

---

## 7) Acceptance Deadlock Guard (Mandatory)

1. Acceptance decision uses a single hard gate first:
   - primary: `Success@LIBERO`
   - constraints: `p95 latency` and stability (`nonfinite/crash`)
   - proxy diagnostics (`loss/mask/teacher/delta`) are supporting evidence, not final pass/fail gate.
2. If remote artifacts are missing, do not keep silent `IN_PROGRESS` forever:
   - mark status as `BLOCKED_WAIT_REMOTE`
   - list missing artifacts explicitly (`run_identity/config/metrics/summary/log`)
   - list owner and next retry time.
3. For repeated entries with same `EXP_ID`, latest entry status is authoritative for triage.
4. If waiting remote evidence exceeds one working day, produce a fallback package:
   - local code diff + command evidence + explicit unblock checklist for user.
5. Do not open next phase gate while current gate is `BLOCKED_WAIT_REMOTE` without an explicit user-approved override.
