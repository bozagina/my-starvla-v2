#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/bazinga/code/my-starvla-v2"
PREFLIGHT="$ROOT/docs/starvla_retrofit/skills/starvla-retrofit-ops/scripts/preflight.sh"
DEADLOCK_CHECK="$ROOT/tools/handoff/check_deadlock_risk.py"
MAX_OPEN_HOURS="${MAX_OPEN_HOURS:-24}"

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
printf "max_open_hours=%s\n" "$MAX_OPEN_HOURS"

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
  if python "$DEADLOCK_CHECK" --max-open-hours "$MAX_OPEN_HOURS"; then
    pass "deadlock risk check passed"
  else
    fail "deadlock risk check failed"
  fi
else
  fail "deadlock checker missing: $DEADLOCK_CHECK"
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
