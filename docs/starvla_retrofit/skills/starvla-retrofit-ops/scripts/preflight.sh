#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${STARVLA_REPO_ROOT:-}" ]]; then
  ROOT="$(cd "${STARVLA_REPO_ROOT}" && pwd -P)"
elif git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT="$(cd "$git_root" && pwd -P)"
else
  ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd -P)"
fi

GUARD="$ROOT/tools/handoff/ensure_repo_context.sh"
guard_args=(--repo-root "$ROOT" --require-expected-root)
if [[ -n "${STARVLA_EXPECTED_REPO_ROOT:-}" ]]; then
  guard_args+=(--expect-root "$STARVLA_EXPECTED_REPO_ROOT")
fi
if [[ -n "${STARVLA_EXPECTED_VLM_SCOPE:-}" ]]; then
  guard_args+=(--expect-vlm-scope "$STARVLA_EXPECTED_VLM_SCOPE")
fi

if [[ ! -f "$GUARD" ]]; then
  echo "[preflight][FAIL] repo guard missing: $GUARD" >&2
  exit 2
fi

if ! bash "$GUARD" "${guard_args[@]}"; then
  echo "[preflight][FAIL] repo guard failed" >&2
  exit 2
fi

CURRENT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"

WORKTREE_BRANCH="${STARVLA_WORKTREE_BRANCH:-}"
if [[ -z "$WORKTREE_BRANCH" ]]; then
  if [[ "$CURRENT_BRANCH" == codex/worktree-* ]]; then
    WORKTREE_BRANCH="$CURRENT_BRANCH"
  else
    WORKTREE_BRANCH="$(git -C "$ROOT" for-each-ref --format='%(refname:short)' refs/heads/codex/worktree-* | sort | tail -n 1 || true)"
  fi
fi

FINAL_BRANCH="${STARVLA_FINAL_BRANCH:-}"
if [[ -z "$FINAL_BRANCH" ]]; then
  FINAL_BRANCH="$(git -C "$ROOT" for-each-ref --format='%(refname:short)' refs/heads/codex/* \
    | grep -E -v '^codex/(tmp-|worktree-)' \
    | sort \
    | tail -n 1 || true)"
fi

printf "[preflight] repo=%s\n" "$ROOT"
printf "[preflight] branch=%s\n" "$CURRENT_BRANCH"
printf "[preflight] expected_repo=%s\n" "${STARVLA_EXPECTED_REPO_ROOT:-<unset>}"
printf "[preflight] expected_vlm_scope=%s\n" "${STARVLA_EXPECTED_VLM_SCOPE:-<unset>}"

printf "[preflight] status:\n"
git -C "$ROOT" status --short --branch

TMP_BRANCH=""
if [[ "$CURRENT_BRANCH" == codex/tmp-* ]]; then
  TMP_BRANCH="$CURRENT_BRANCH"
else
  # fallback to latest tmp branch in repo when current branch is not tmp.
  TMP_BRANCH="$(git -C "$ROOT" for-each-ref --format='%(refname:short)' refs/heads/codex/tmp-* | sort | tail -n 1 || true)"
fi

printf "[preflight] branch topology check:\n"
if [[ -n "$TMP_BRANCH" ]]; then
  printf "  - %s : OK (resolved tmp branch)\n" "$TMP_BRANCH"
else
  printf "  - codex/tmp-* : MISSING\n"
fi

for b in "$WORKTREE_BRANCH" "$FINAL_BRANCH"; do
  if [[ -z "$b" ]]; then
    continue
  fi
  if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$b"; then
    printf "  - %s : OK\n" "$b"
  else
    printf "  - %s : MISSING\n" "$b"
  fi
done

if [[ -n "$TMP_BRANCH" ]] && [[ -n "$WORKTREE_BRANCH" ]] && git -C "$ROOT" show-ref --verify --quiet "refs/heads/$WORKTREE_BRANCH"; then
  if git -C "$ROOT" merge-base --is-ancestor "$WORKTREE_BRANCH" "$TMP_BRANCH"; then
    printf "[preflight] ancestry worktree->tmp: OK\n"
  else
    printf "[preflight] ancestry worktree->tmp: FAIL\n"
  fi
else
  printf "[preflight] ancestry worktree->tmp: SKIP (missing tmp/worktree)\n"
fi

if [[ -n "$WORKTREE_BRANCH" ]] && [[ -n "$FINAL_BRANCH" ]] \
  && git -C "$ROOT" show-ref --verify --quiet "refs/heads/$WORKTREE_BRANCH" \
  && git -C "$ROOT" show-ref --verify --quiet "refs/heads/$FINAL_BRANCH" \
  && git -C "$ROOT" merge-base --is-ancestor "$WORKTREE_BRANCH" "$FINAL_BRANCH"; then
  printf "[preflight] ancestry worktree->final: OK\n"
else
  if [[ -n "$WORKTREE_BRANCH" ]] && [[ -n "$FINAL_BRANCH" ]] \
    && git -C "$ROOT" show-ref --verify --quiet "refs/heads/$WORKTREE_BRANCH" \
    && git -C "$ROOT" show-ref --verify --quiet "refs/heads/$FINAL_BRANCH"; then
    printf "[preflight] ancestry worktree->final: FAIL\n"
  else
    printf "[preflight] ancestry worktree->final: SKIP (missing worktree/final)\n"
  fi
fi
