#!/usr/bin/env python3
"""Check progress log for deadlock-prone status patterns."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESS = ROOT / "docs/algorithm1/handoff/progress_live.md"
EXP_RE = re.compile(r"(ALG1-[A-Z]+-\d{8}-\d{3}-[A-Z0-9_]+)")
HEADER_RE = re.compile(r"^## \[([^\]]+)\] (.+)$")
STATUS_RE = re.compile(r"^- Status:\s*([A-Z_]+)\s*$", re.MULTILINE)
OPEN_STATUSES = {"IN_PROGRESS", "BLOCKED_WAIT_REMOTE"}


def _parse_entries(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    entries: List[Dict[str, str]] = []
    i = 0
    while i < len(lines):
        m = HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        ts_text = m.group(1).strip()
        title = m.group(2).strip()
        body_lines: List[str] = []
        i += 1
        while i < len(lines) and not HEADER_RE.match(lines[i]):
            body_lines.append(lines[i])
            i += 1
        exp_m = EXP_RE.search(title)
        if not exp_m:
            continue
        status_m = STATUS_RE.search("\n".join(body_lines))
        if not status_m:
            continue
        entries.append(
            {
                "exp_id": exp_m.group(1),
                "timestamp": ts_text,
                "title": title,
                "status": status_m.group(1).strip().upper(),
            }
        )
    return entries


def _parse_ts(ts_text: str) -> dt.datetime:
    # format expected: YYYY-MM-DD HH:MM:SS +08:00
    return dt.datetime.fromisoformat(ts_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect deadlock risks from progress log statuses.")
    parser.add_argument("--progress-file", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--max-open-hours", type=float, default=24.0)
    args = parser.parse_args()

    text = args.progress_file.read_text(encoding="utf-8")
    entries = _parse_entries(text)
    if not entries:
        print("No EXP status entries found.")
        return 1

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for e in entries:
        grouped[e["exp_id"]].append(e)

    now = dt.datetime.now().astimezone()
    stale_open: List[Tuple[str, str, float]] = []
    reopened_after_done: List[str] = []

    for exp_id, seq in grouped.items():
        latest = seq[-1]
        statuses = [x["status"] for x in seq]
        if "DONE" in statuses and latest["status"] in OPEN_STATUSES:
            reopened_after_done.append(exp_id)
        if latest["status"] in OPEN_STATUSES:
            try:
                age_h = (now - _parse_ts(latest["timestamp"])).total_seconds() / 3600.0
            except Exception:
                age_h = -1.0
            if age_h >= args.max_open_hours:
                stale_open.append((exp_id, latest["status"], age_h))

    print("=== Deadlock Risk Report ===")
    print(f"progress_file={args.progress_file}")
    print(f"exp_entries={len(entries)} exp_ids={len(grouped)}")
    print(f"max_open_hours={args.max_open_hours}")

    print("\nLatest Open Entries (stale):")
    if not stale_open:
        print("- none")
    else:
        for exp_id, status, age_h in sorted(stale_open):
            print(f"- {exp_id}: {status}, age_hours={age_h:.1f}")

    print("\nStatus Reopen After DONE:")
    if not reopened_after_done:
        print("- none")
    else:
        for exp_id in sorted(set(reopened_after_done)):
            print(f"- {exp_id}")

    if stale_open or reopened_after_done:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
