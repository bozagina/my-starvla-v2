#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${STARVLA_REPO_ROOT:-}" ]]; then
  ROOT="$(cd "${STARVLA_REPO_ROOT}" && pwd -P)"
elif git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT="$(cd "$git_root" && pwd -P)"
else
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
fi

REPO_GUARD="$ROOT/tools/handoff/ensure_repo_context.sh"
PREFLIGHT="$ROOT/docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh"
DEADLOCK_CHECK="$ROOT/tools/handoff/check_deadlock_risk.py"
MAX_OPEN_HOURS="${MAX_OPEN_HOURS:-24}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN=""
fi

REQUIRED_FILES=(
  "$ROOT/docs/starvla_retrofit/handoff/context_pack_compact.md"
  "$ROOT/docs/starvla_retrofit/handoff/system_prompt_operating_contract.md"
  "$ROOT/docs/starvla_retrofit/handoff/acceptance_deadlock_guard.md"
  "$ROOT/docs/starvla_retrofit/handoff/retrofit_prompt_and_branch_strategy.md"
  "$ROOT/docs/starvla_retrofit/handoff/star_vla改造与统一伪标签生成任务书.md"
  "$ROOT/docs/algorithm1/handoff/progress_live.md"
)

overall_fail=0
warn_count=0

pass() { printf "[PASS] %s\n" "$1"; }
warn() { printf "[WARN] %s\n" "$1"; warn_count=$((warn_count + 1)); }
fail() { printf "[FAIL] %s\n" "$1"; overall_fail=1; }

printf "=== Pre-Dev Readiness ===\n"
printf "repo=%s\n" "$ROOT"
printf "expected_repo=%s\n" "${STARVLA_EXPECTED_REPO_ROOT:-<unset>}"
printf "expected_vlm_scope=%s\n" "${STARVLA_EXPECTED_VLM_SCOPE:-<unset>}"
printf "max_open_hours=%s\n" "$MAX_OPEN_HOURS"

guard_args=(--repo-root "$ROOT" --require-expected-root)
if [[ -n "${STARVLA_EXPECTED_REPO_ROOT:-}" ]]; then
  guard_args+=(--expect-root "$STARVLA_EXPECTED_REPO_ROOT")
fi
if [[ -n "${STARVLA_EXPECTED_VLM_SCOPE:-}" ]]; then
  guard_args+=(--expect-vlm-scope "$STARVLA_EXPECTED_VLM_SCOPE")
fi

if [[ -f "$REPO_GUARD" ]]; then
  if bash "$REPO_GUARD" "${guard_args[@]}"; then
    pass "repo context guard passed"
  else
    fail "repo context guard failed"
  fi
else
  fail "repo guard missing: $REPO_GUARD"
fi

for f in "${REQUIRED_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    pass "required file exists: $f"
  else
    fail "required file missing: $f"
  fi
done

if [[ -x "$PREFLIGHT" || -f "$PREFLIGHT" ]]; then
  if bash "$PREFLIGHT"; then
    pass "preflight topology check passed"
  else
    fail "preflight topology check failed"
  fi
else
  fail "preflight script missing: $PREFLIGHT"
fi

if [[ -f "$DEADLOCK_CHECK" ]]; then
  if [[ -z "$PYTHON_BIN" ]]; then
    fail "python interpreter not found (need python or python3)"
  else
    if "$PYTHON_BIN" "$DEADLOCK_CHECK" --max-open-hours "$MAX_OPEN_HOURS"; then
      pass "deadlock risk check passed"
    else
      deadlock_code=$?
      if [[ "$deadlock_code" -eq 1 ]]; then
        warn "deadlock risk check has no EXP entries yet; initialize progress log via bootstrap_session.sh start"
      else
        fail "deadlock risk check failed"
      fi
    fi
  fi
else
  fail "deadlock checker missing: $DEADLOCK_CHECK"
fi

THREAD_VALIDATOR="$ROOT/tools/handoff/validate_thread_v2.py"
THREAD_ID="${STARVLA_THREAD_ID:-}"

if [[ -n "$THREAD_ID" ]]; then
  printf "\n=== Harness v2 Thread Checks (thread=%s) ===\n" "$THREAD_ID"

  case "$THREAD_ID" in
    A-RES|A-BUILD|A-REVIEW)
      thread_manifest="$ROOT/docs/algorithm1/handoff/manifests/a_module_current_round.yaml"
      ;;
    CP-RES|CP-BUILD|CP-REVIEW)
      thread_manifest="$ROOT/docs/algorithm1/handoff/manifests/cp_current_round.yaml"
      ;;
    *)
      fail "invalid STARVLA_THREAD_ID: $THREAD_ID (expected A-RES|A-BUILD|A-REVIEW|CP-RES|CP-BUILD|CP-REVIEW)"
      thread_manifest=""
      ;;
  esac

  if [[ -n "$thread_manifest" ]]; then
    if [[ -f "$thread_manifest" ]]; then
      pass "thread manifest exists: $thread_manifest"
    else
      fail "thread manifest missing: $thread_manifest"
    fi
  fi

  if [[ -f "$THREAD_VALIDATOR" && -n "$PYTHON_BIN" ]]; then
    validator_output=""
    if validator_output=$("$PYTHON_BIN" "$THREAD_VALIDATOR" --thread-id "$THREAD_ID" --repo-root "$ROOT" 2>&1); then
      pass "thread validator passed"
    else
      fail "thread validator failed"
      printf "%s\n" "$validator_output" >&2
    fi
  elif [[ ! -f "$THREAD_VALIDATOR" ]]; then
    warn "thread validator not found: $THREAD_VALIDATOR"
  elif [[ -z "$PYTHON_BIN" ]]; then
    warn "python interpreter not found for thread validator"
  fi
else
  printf "\n=== Harness v2 Thread Checks (skipped: STARVLA_THREAD_ID unset) ===\n"
  warn "STARVLA_THREAD_ID is not set; v2 thread checks skipped. Set to one of: A-RES A-BUILD A-REVIEW CP-RES CP-BUILD CP-REVIEW"
fi

current_branch="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" == codex/tmp-* ]]; then
  pass "current branch is tmp branch: $current_branch"
else
  warn "current branch is not codex/tmp-*: $current_branch"
fi

status_lines="$(git -C "$ROOT" status --short | wc -l | tr -d ' ')"
if [[ "$status_lines" == "0" ]]; then
  pass "working tree is clean"
else
  warn "working tree is dirty ($status_lines paths); ensure unrelated files are not mixed into next patch"
fi

if [[ "$overall_fail" -eq 0 ]]; then
  printf "READY_TO_DEVELOP=YES\n"
  if [[ "$warn_count" -gt 0 ]]; then
    printf "READINESS_NOTES=%s warnings\n" "$warn_count"
  fi
  exit 0
fi

printf "READY_TO_DEVELOP=NO\n"
exit 2
