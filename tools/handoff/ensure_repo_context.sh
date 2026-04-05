#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ensure_repo_context.sh [--repo-root <PATH>] [--expect-root <PATH>] [--expect-vlm-scope <qwen_only|mixed|auto>] [--require-expected-root]

Env:
  STARVLA_REPO_ROOT             Optional repo root override.
  STARVLA_EXPECTED_REPO_ROOT    Expected git root for this session.
  STARVLA_EXPECTED_VLM_SCOPE    Expected VLM scope: qwen_only|mixed|auto.

Examples:
  bash tools/handoff/ensure_repo_context.sh \
    --expect-root /2025233147/zzq/SpatialVLA_llava3d/starvla_test_qwen/starVLA \
    --expect-vlm-scope qwen_only \
    --require-expected-root
EOF
}

canonical_dir() {
  local path="$1"
  if [[ -d "$path" ]]; then
    (cd "$path" && pwd -P)
    return
  fi

  local parent
  parent="$(dirname "$path")"
  if [[ -d "$parent" ]]; then
    printf "%s/%s\n" "$(cd "$parent" && pwd -P)" "$(basename "$path")"
    return
  fi

  printf "%s\n" "$path"
}

resolve_repo_root() {
  local fallback_script_dir="$1"
  if [[ -n "${REPO_ROOT_ARG:-}" ]]; then
    canonical_dir "$REPO_ROOT_ARG"
    return
  fi
  if [[ -n "${STARVLA_REPO_ROOT:-}" ]]; then
    canonical_dir "$STARVLA_REPO_ROOT"
    return
  fi
  if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    canonical_dir "$git_root"
    return
  fi
  canonical_dir "$(cd "$fallback_script_dir/../.." && pwd -P)"
}

REPO_ROOT_ARG=""
EXPECT_ROOT_ARG=""
EXPECT_VLM_SCOPE_ARG=""
REQUIRE_EXPECTED_ROOT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT_ARG="${2:-}"
      shift 2
      ;;
    --expect-root)
      EXPECT_ROOT_ARG="${2:-}"
      shift 2
      ;;
    --expect-vlm-scope)
      EXPECT_VLM_SCOPE_ARG="${2:-}"
      shift 2
      ;;
    --require-expected-root)
      REQUIRE_EXPECTED_ROOT=1
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "[repo-guard][FAIL] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(resolve_repo_root "$SCRIPT_DIR")"
EXPECT_ROOT="${EXPECT_ROOT_ARG:-${STARVLA_EXPECTED_REPO_ROOT:-}}"
EXPECT_VLM_SCOPE="${EXPECT_VLM_SCOPE_ARG:-${STARVLA_EXPECTED_VLM_SCOPE:-auto}}"

echo "[repo-guard] repo_root=$REPO_ROOT"
echo "[repo-guard] expected_root=${EXPECT_ROOT:-<unset>}"
echo "[repo-guard] expected_vlm_scope=$EXPECT_VLM_SCOPE"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[repo-guard][FAIL] repo root does not exist: $REPO_ROOT" >&2
  echo "REPO_CONTEXT_OK=NO"
  exit 2
fi

if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[repo-guard][FAIL] not a git worktree: $REPO_ROOT" >&2
  echo "REPO_CONTEXT_OK=NO"
  exit 2
fi

if [[ "$REQUIRE_EXPECTED_ROOT" -eq 1 && -z "$EXPECT_ROOT" ]]; then
  echo "[repo-guard][FAIL] expected root is required but unset (set STARVLA_EXPECTED_REPO_ROOT or pass --expect-root)." >&2
  echo "REPO_CONTEXT_OK=NO"
  exit 2
fi

if [[ -n "$EXPECT_ROOT" ]]; then
  EXPECT_ROOT_CANON="$(canonical_dir "$EXPECT_ROOT")"
  if [[ ! -d "$EXPECT_ROOT_CANON" ]]; then
    echo "[repo-guard][FAIL] expected root path not found: $EXPECT_ROOT_CANON" >&2
    echo "REPO_CONTEXT_OK=NO"
    exit 2
  fi
  if [[ "$REPO_ROOT" != "$EXPECT_ROOT_CANON" ]]; then
    echo "[repo-guard][FAIL] repo root mismatch." >&2
    echo "[repo-guard][FAIL] expected: $EXPECT_ROOT_CANON" >&2
    echo "[repo-guard][FAIL] actual:   $REPO_ROOT" >&2
    echo "REPO_CONTEXT_OK=NO"
    exit 2
  fi
fi

required_paths=(
  "$REPO_ROOT/starVLA/training/train_starvla.py"
  "$REPO_ROOT/tools/handoff/bootstrap_session.sh"
)
for p in "${required_paths[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "[repo-guard][FAIL] missing required file: $p" >&2
    echo "REPO_CONTEXT_OK=NO"
    exit 2
  fi
done

case "$EXPECT_VLM_SCOPE" in
  qwen_only)
    forbidden_hits="$(
      rg -n -m 5 "(LLaVA_3D|MapAnythingLlava3DPI|mapanything_llava3d|from[[:space:]]+LLaVA_3D)" \
        "$REPO_ROOT/starVLA" "$REPO_ROOT/examples" \
        -g '*.py' -g '*.sh' 2>/dev/null || true
    )"
    if [[ -n "$forbidden_hits" ]]; then
      echo "[repo-guard][FAIL] qwen_only scope violated by runtime code references:" >&2
      echo "$forbidden_hits" >&2
      echo "REPO_CONTEXT_OK=NO"
      exit 2
    fi
    ;;
  mixed|auto)
    :
    ;;
  *)
    echo "[repo-guard][FAIL] invalid --expect-vlm-scope value: $EXPECT_VLM_SCOPE" >&2
    echo "REPO_CONTEXT_OK=NO"
    exit 2
    ;;
esac

echo "REPO_CONTEXT_OK=YES"
