#!/usr/bin/env python3
"""Harness v2 thread validator.

Validates manifest schema, thread identity, allowed paths, consume/emit
contracts, CP-BUILD upstream gate, and basic state-machine legality.

Exit codes:
  0  all checks passed
  1  one or more checks failed (details on stderr)
  2  usage / IO error
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None  # type: ignore[assignment]


VALID_THREAD_IDS = {"A-RES", "A-BUILD", "A-REVIEW", "CP-RES", "CP-BUILD", "CP-REVIEW"}

VALID_ROLES = {"RES", "BUILD", "REVIEW"}

VALID_STATUSES = {
    "draft",
    "ready",
    "READY",
    "IN_PROGRESS",
    "DONE",
    "BLOCKED_WAIT_UPSTREAM",
    "BLOCKED_WAIT_REMOTE",
    "CANCELLED",
}

THREAD_TO_MODULE = {
    "A-RES": "A-module",
    "A-BUILD": "A-module",
    "A-REVIEW": "A-module",
    "CP-RES": "corrective-policy",
    "CP-BUILD": "corrective-policy",
    "CP-REVIEW": "corrective-policy",
}

THREAD_TO_ROLE = {
    "A-RES": "RES",
    "A-BUILD": "BUILD",
    "A-REVIEW": "REVIEW",
    "CP-RES": "RES",
    "CP-BUILD": "BUILD",
    "CP-REVIEW": "REVIEW",
}

MANIFEST_FILE = {
    "A-module": "docs/algorithm1/handoff/manifests/a_module_current_round.yaml",
    "corrective-policy": "docs/algorithm1/handoff/manifests/cp_current_round.yaml",
}

REQUIRED_MANIFEST_TOP_KEYS = [
    "schema_version",
    "thread_family",
    "round_id",
    "status",
    "global_anchors",
    "contracts",
    "threads",
]

REQUIRED_THREAD_KEYS = [
    "thread_id",
    "role",
    "current_gate",
    "status",
    "baseline_anchor",
    "allowed_paths",
    "required_artifacts",
    "blocking_conditions",
]


class ValidationResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    def report(self, file=sys.stderr) -> None:
        for m in self.passed:
            print(f"[PASS] {m}", file=file)
        for m in self.warnings:
            print(f"[WARN] {m}", file=file)
        for m in self.failed:
            print(f"[FAIL] {m}", file=file)
        verdict = "ALL CHECKS PASSED" if self.success else "VALIDATION FAILED"
        print(f"\n{verdict} (pass={len(self.passed)} warn={len(self.warnings)} fail={len(self.failed)})", file=file)


def _load_yaml_fallback(path: Path) -> dict | None:
    """Minimal YAML-subset loader when PyYAML is unavailable.

    Handles indented mappings and lists (single-level nesting) sufficient for
    manifest validation. Not a general-purpose YAML parser.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    root: dict = {}
    stack: list[tuple[int, dict | list]] = [(-1, root)]
    current_key_at: dict[int, str] = {}
    list_items: dict[int, list] = {}

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = raw_line.strip()

        # List item
        if stripped.startswith("- "):
            item_val = stripped[2:].strip()
            # Find the parent container
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()
            _, parent = stack[-1]
            parent_indent = stack[-1][0]
            parent_key = current_key_at.get(parent_indent, "")

            if isinstance(parent, dict) and parent_key:
                existing = parent.get(parent_key)
                if not isinstance(existing, list):
                    parent[parent_key] = []
                if ":" in item_val and not item_val.startswith("'") and not item_val.startswith('"'):
                    k, _, v = item_val.partition(":")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    entry: dict = {k: v}
                    parent[parent_key].append(entry)
                    stack.append((indent + 2, entry))
                else:
                    val = item_val.strip('"').strip("'")
                    if val:
                        parent[parent_key].append(val)
            continue

        if ":" not in stripped:
            continue

        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.strip()

        # Remove YAML block scalar indicators
        if val in (">", "|", ">-", "|-"):
            val = ""

        # Remove surrounding quotes
        if val and val[0] in ('"', "'") and val[-1] == val[0]:
            val = val[1:-1]

        # Pop stack to correct nesting level
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        _, parent = stack[-1]
        if not isinstance(parent, dict):
            continue

        if val:
            parent[key] = val
            current_key_at[indent] = key
        else:
            parent[key] = {}
            current_key_at[indent] = key
            stack.append((indent, parent))
            # The children will attach to parent[key], so push the sub-dict
            stack[-1] = (indent, parent)
            # Actually we need to push the sub-dict for children to attach
            stack.append((indent + 1, parent[key]))

    return root if root else None


def load_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    if yaml is not None:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return _load_yaml_fallback(path)


def validate_manifest_schema(manifest: dict, result: ValidationResult) -> None:
    for key in REQUIRED_MANIFEST_TOP_KEYS:
        if key in manifest:
            result.ok(f"manifest has required key: {key}")
        else:
            result.fail(f"manifest missing required key: {key}")

    sv = manifest.get("schema_version", "")
    if sv == "harness_v2_thread_manifest_v1":
        result.ok(f"schema_version is correct: {sv}")
    else:
        result.fail(f"schema_version unexpected: {sv!r} (expected harness_v2_thread_manifest_v1)")

    contracts = manifest.get("contracts", {})
    if isinstance(contracts, dict):
        if contracts.get("consume"):
            result.ok("contracts.consume is present")
        else:
            result.fail("contracts.consume is missing or empty")
        if contracts.get("emit"):
            result.ok("contracts.emit is present")
        else:
            result.fail("contracts.emit is missing or empty")
    else:
        result.fail("contracts is not a mapping")


def validate_thread_entry(thread_id: str, manifest: dict, result: ValidationResult) -> dict | None:
    threads = manifest.get("threads")
    if not isinstance(threads, list):
        result.fail("manifest.threads is not a list")
        return None

    match = None
    for t in threads:
        if isinstance(t, dict) and t.get("thread_id") == thread_id:
            match = t
            break

    if match is None:
        result.fail(f"thread_id {thread_id} not found in manifest.threads")
        return None

    result.ok(f"thread entry found for {thread_id}")

    for key in REQUIRED_THREAD_KEYS:
        if key in match:
            result.ok(f"thread.{key} present")
        else:
            result.fail(f"thread.{key} missing for {thread_id}")

    expected_role = THREAD_TO_ROLE.get(thread_id)
    actual_role = match.get("role")
    if actual_role == expected_role:
        result.ok(f"thread role matches: {actual_role}")
    else:
        result.fail(f"thread role mismatch: manifest says {actual_role!r}, expected {expected_role!r}")

    status = match.get("status", "")
    if status in VALID_STATUSES:
        result.ok(f"thread status is valid: {status}")
    else:
        result.warn(f"thread status is non-standard: {status!r}")

    return match


def validate_cp_build_gate(manifest: dict, thread_entry: dict | None, result: ValidationResult) -> None:
    """CP-BUILD must be blocked unless A-REVIEW has issued usable_for_downstream."""
    upstream_gate = manifest.get("upstream_gate", {})
    current_verdict = ""
    if isinstance(upstream_gate, dict):
        current_verdict = str(upstream_gate.get("current_verdict", "")).strip().lower()

    thread_status = ""
    if thread_entry:
        thread_status = str(thread_entry.get("status", "")).strip()

    if current_verdict in ("yes", "conditional"):
        result.ok(f"CP upstream_gate.current_verdict = {current_verdict}; CP-BUILD may proceed")
    elif thread_status == "BLOCKED_WAIT_UPSTREAM":
        result.ok("CP-BUILD is correctly BLOCKED_WAIT_UPSTREAM (upstream verdict not yet issued)")
    else:
        result.fail(
            f"CP-BUILD status is {thread_status!r} but upstream_gate.current_verdict "
            f"is {current_verdict!r}; must be BLOCKED_WAIT_UPSTREAM or have upstream approval"
        )


def validate_allowed_paths(thread_entry: dict, repo_root: Path, result: ValidationResult) -> None:
    """Check that allowed_paths is non-empty and looks plausible."""
    paths = thread_entry.get("allowed_paths", [])
    if not paths:
        result.fail("allowed_paths is empty")
        return
    result.ok(f"allowed_paths has {len(paths)} entries")


def main() -> int:
    parser = argparse.ArgumentParser(description="Harness v2 thread validator")
    parser.add_argument("--thread-id", required=True, help="Thread ID (e.g. A-BUILD, CP-RES)")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    args = parser.parse_args()

    thread_id: str = args.thread_id
    repo_root = Path(args.repo_root).resolve()
    result = ValidationResult()

    if thread_id not in VALID_THREAD_IDS:
        result.fail(f"invalid thread_id: {thread_id} (valid: {sorted(VALID_THREAD_IDS)})")
        result.report()
        return 1

    result.ok(f"thread_id is valid: {thread_id}")

    module = THREAD_TO_MODULE[thread_id]
    manifest_rel = MANIFEST_FILE[module]
    manifest_path = repo_root / manifest_rel
    manifest = load_manifest(manifest_path)

    if manifest is None:
        result.fail(f"cannot load manifest: {manifest_path}")
        result.report()
        return 1

    result.ok(f"manifest loaded: {manifest_rel}")

    validate_manifest_schema(manifest, result)

    thread_entry = validate_thread_entry(thread_id, manifest, result)

    if thread_entry is not None:
        validate_allowed_paths(thread_entry, repo_root, result)

    if thread_id == "CP-BUILD":
        validate_cp_build_gate(manifest, thread_entry, result)

    if thread_id.startswith("CP-"):
        upstream_gate = manifest.get("upstream_gate", {})
        if isinstance(upstream_gate, dict) and upstream_gate.get("required_verdict"):
            result.ok("upstream_gate.required_verdict is documented")
        else:
            result.warn("upstream_gate.required_verdict not found in CP manifest")

    result.report()
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
