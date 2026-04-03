#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/bazinga/code/my-starvla-v2"
WORKTREE_BRANCH="codex/worktree-starvla-v2-mainline"
FINAL_BRANCH="codex/starvla-retrofit-unified-pseudolabel-20260402"
CURRENT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"

printf "[preflight] repo=%s\n" "$ROOT"
printf "[preflight] branch=%s\n" "$CURRENT_BRANCH"

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
  if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$b"; then
    printf "  - %s : OK\n" "$b"
  else
    printf "  - %s : MISSING\n" "$b"
  fi
done

if [[ -n "$TMP_BRANCH" ]] && git -C "$ROOT" show-ref --verify --quiet "refs/heads/$WORKTREE_BRANCH"; then
  if git -C "$ROOT" merge-base --is-ancestor "$WORKTREE_BRANCH" "$TMP_BRANCH"; then
    printf "[preflight] ancestry worktree->tmp: OK\n"
  else
    printf "[preflight] ancestry worktree->tmp: FAIL\n"
  fi
else
  printf "[preflight] ancestry worktree->tmp: SKIP (missing tmp/worktree)\n"
fi

if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$WORKTREE_BRANCH" \
  && git -C "$ROOT" show-ref --verify --quiet "refs/heads/$FINAL_BRANCH" \
  && git -C "$ROOT" merge-base --is-ancestor "$WORKTREE_BRANCH" "$FINAL_BRANCH"; then
  printf "[preflight] ancestry worktree->final: OK\n"
else
  if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$WORKTREE_BRANCH" \
    && git -C "$ROOT" show-ref --verify --quiet "refs/heads/$FINAL_BRANCH"; then
    printf "[preflight] ancestry worktree->final: FAIL\n"
  else
    printf "[preflight] ancestry worktree->final: SKIP (missing worktree/final)\n"
  fi
fi
