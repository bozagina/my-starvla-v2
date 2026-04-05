#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEGACY_PROMPT_DOC="$ROOT/docs/algorithm1/handoff/new_chat_bootstrap_command.md"
RETROFIT_PROMPT_DOC="$ROOT/docs/starvla_retrofit/handoff/new_chat_bootstrap_compact.md"
PROMPT_DOC="$RETROFIT_PROMPT_DOC"
PROGRESS_TOOL="$ROOT/tools/handoff/new_progress_entry.py"

usage() {
  cat <<'EOF'
Usage:
  bootstrap_session.sh prompt
      Print the default startup instruction (retrofit compact flow).

  bootstrap_session.sh prompt-retrofit
      Print the compact retrofit startup instruction for a new chat.

  bootstrap_session.sh prompt-legacy
      Print the legacy algorithm1 startup instruction.

  bootstrap_session.sh start --module <MASK|FBLOSS|CFG|DIAG|DATA|INFRA|EVAL> [--owner <TAG>] [--title <TEXT>]
      Create a standardized IN_PROGRESS progress entry (EXP_ID) for this session.

  bootstrap_session.sh help
EOF
}

print_prompt() {
  local prompt_doc="${1:-$PROMPT_DOC}"
  if [[ ! -f "$prompt_doc" ]]; then
    echo "Prompt doc not found: $prompt_doc" >&2
    exit 1
  fi
  awk '/^```text$/{flag=1;next}/^```$/{if(flag){exit}}flag' "$prompt_doc"
}

start_session() {
  local module=""
  local owner="OC"
  local title="New session task"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --module)
        module="${2:-}"
        shift 2
        ;;
      --owner)
        owner="${2:-}"
        shift 2
        ;;
      --title)
        title="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "$module" ]]; then
    echo "Missing required --module" >&2
    usage
    exit 1
  fi

  python "$PROGRESS_TOOL" --module "$module" --owner "$owner" --title "$title"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  prompt)
    print_prompt "$PROMPT_DOC"
    ;;
  prompt-retrofit)
    print_prompt "$RETROFIT_PROMPT_DOC"
    ;;
  prompt-legacy)
    print_prompt "$LEGACY_PROMPT_DOC"
    ;;
  start)
    start_session "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
