# Acceptance Deadlock Guard (Dev <-> Acceptance)

Purpose: prevent deadlock where acceptance waits for artifacts that development thread cannot provide yet.

---

## 1) Handshake Contract

Acceptance can request only this minimum set:

1. `run_identity.txt`
2. `config.yaml`
3. `metrics.jsonl`
4. `summary.jsonl`
5. `train.log` or `train.raw.log`

If any item is missing, development must not stay in silent `IN_PROGRESS`.

---

## 2) Status Machine

Use these statuses in `progress_live.md`:

1. `IN_PROGRESS`: local work is still advancing.
2. `BLOCKED_WAIT_REMOTE`: blocked on remote artifacts or remote rerun.
3. `BLOCKED`: blocked by local dependency not related to remote artifacts.
4. `DONE`: acceptance criteria reached for current gate.
5. `ROLLED_BACK`: reverted to stable baseline after failed attempt.

Rule: for repeated entries with same `EXP_ID`, latest status is authoritative.

---

## 3) Single Acceptance Arbitration

Use one hard gate first:

1. primary outcome: `Success@LIBERO`
2. hard constraints: `p95 latency` and stability (`nonfinite/crash`)
3. proxy diagnostics (`loss/mask/teacher/delta`) are supporting evidence only.

This prevents train-proxy metrics from overriding rollout acceptance.

---

## 4) Timeout and Escalation

If waiting for remote evidence exceeds one working day:

1. append `BLOCKED_WAIT_REMOTE` entry with owner and retry time
2. include exact missing artifact checklist
3. publish fallback package:
   - local diff summary
   - commands already executed
   - expected remote launch command/config path

---

## 5) Operational Commands

Lock repo context first:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export STARVLA_EXPECTED_REPO_ROOT="<absolute-target-repo-root>"
export STARVLA_EXPECTED_VLM_SCOPE="qwen_only"
bash "$REPO_ROOT/tools/handoff/ensure_repo_context.sh" --expect-root "$STARVLA_EXPECTED_REPO_ROOT" --expect-vlm-scope "$STARVLA_EXPECTED_VLM_SCOPE" --require-expected-root
```

Preflight:

```bash
bash <REPO_ROOT>/docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh
```

Create session entry:

```bash
<REPO_ROOT>/tools/handoff/bootstrap_session.sh start \
  --module INFRA \
  --owner OC \
  --title "<task title>"
```

Fetch artifacts:

```bash
bash <REPO_ROOT>/tools/fetch_latest_run_files.sh
```

Check deadlock risk from progress statuses:

```bash
python <REPO_ROOT>/tools/handoff/check_deadlock_risk.py --max-open-hours 24
```

One-shot readiness check before starting development:

```bash
bash <REPO_ROOT>/tools/handoff/pre_dev_readiness.sh
```
