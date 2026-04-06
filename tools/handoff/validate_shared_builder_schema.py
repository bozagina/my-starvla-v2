#!/usr/bin/env python3
"""Validate shared-builder smoke JSON blocks from log files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _extract_blocks(text: str) -> list[dict]:
    pattern = re.compile(r"SMOKE_RESULT_JSON_START\s*(\{.*?\})\s*SMOKE_RESULT_JSON_END", re.S)
    blocks: list[dict] = []
    for match in pattern.finditer(text):
        blocks.append(json.loads(match.group(1)))
    return blocks


def _shape_2d(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        t, d = int(value[0]), int(value[1])
    except Exception:
        return None
    if t <= 0 or d <= 0:
        return None
    return [t, d]


def _validate_one(
    block: dict,
    mode: str,
    expected_schema_version: str,
    expected_action_chunk_len: int | None,
    expected_action_dim: int | None,
    require_state: bool,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(block, dict):
        return ["smoke block must be dict"]

    sample_keys = set(block.get("sample_keys") or [])
    action_shape = _shape_2d(block.get("action_shape"))
    chunk_shape_raw = block.get("action_chunk_shape")
    chunk_shape = _shape_2d(chunk_shape_raw) if chunk_shape_raw is not None else None
    meta = block.get("meta")

    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "shared" if {"action_chunk", "meta", "obs"}.issubset(sample_keys) else "legacy"

    if action_shape is None:
        errors.append("`action_shape` must be [T,D] with positive dims.")
    else:
        if expected_action_chunk_len is not None and action_shape[0] != expected_action_chunk_len:
            errors.append(
                f"`action_shape[0]` mismatch: got {action_shape[0]}, expected {expected_action_chunk_len}"
            )
        if expected_action_dim is not None and action_shape[1] != expected_action_dim:
            errors.append(f"`action_shape[1]` mismatch: got {action_shape[1]}, expected {expected_action_dim}")

    if require_state and block.get("state_shape") is None:
        errors.append("`state_shape` missing while `require_state=true`.")

    if resolved_mode == "shared":
        missing = sorted({"action_chunk", "meta", "obs"} - sample_keys)
        if missing:
            errors.append(f"shared mode missing keys: {missing}")
        if chunk_shape is None:
            errors.append("shared mode expects `action_chunk_shape=[T,D]`.")
        elif action_shape is not None and chunk_shape != action_shape:
            errors.append(f"`action_chunk_shape` mismatch: got {chunk_shape}, expected {action_shape}")
        if not isinstance(meta, dict):
            errors.append("shared mode requires `meta` dict.")
        else:
            if meta.get("schema_version") != expected_schema_version:
                errors.append(
                    f"`meta.schema_version` mismatch: got {meta.get('schema_version')!r}, "
                    f"expected {expected_schema_version!r}"
                )
            for key in ["action_chunk_len", "action_dim", "dataset_name", "trajectory_id", "sample_step"]:
                if key not in meta:
                    errors.append(f"`meta.{key}` missing.")
            if action_shape is not None:
                if meta.get("action_chunk_len") != action_shape[0]:
                    errors.append(
                        f"`meta.action_chunk_len` mismatch: got {meta.get('action_chunk_len')}, expected {action_shape[0]}"
                    )
                if meta.get("action_dim") != action_shape[1]:
                    errors.append(
                        f"`meta.action_dim` mismatch: got {meta.get('action_dim')}, expected {action_shape[1]}"
                    )
    else:
        if "action_chunk" in sample_keys or meta is not None:
            errors.append("legacy mode must not contain shared-only fields (`action_chunk`/`meta`).")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-log", required=True)
    parser.add_argument("--mode", choices=["auto", "shared", "legacy"], default="auto")
    parser.add_argument("--expected-schema-version", default="p1_shared_builder_v1")
    parser.add_argument("--expected-action-chunk-len", type=int, default=None)
    parser.add_argument("--expected-action-dim", type=int, default=None)
    parser.add_argument("--require-state", action="store_true")
    args = parser.parse_args()

    text = Path(args.smoke_log).read_text(encoding="utf-8", errors="replace")
    blocks = _extract_blocks(text)
    if not blocks:
        print(f"[FAIL] no smoke json blocks found in {args.smoke_log}")
        return 2

    all_errors: list[str] = []
    for i, block in enumerate(blocks):
        errs = _validate_one(
            block=block,
            mode=args.mode,
            expected_schema_version=args.expected_schema_version,
            expected_action_chunk_len=args.expected_action_chunk_len,
            expected_action_dim=args.expected_action_dim,
            require_state=args.require_state,
        )
        if errs:
            for err in errs:
                all_errors.append(f"block[{i}] {err}")

    if all_errors:
        print("[FAIL] shared builder schema validation failed:")
        for err in all_errors:
            print(f"- {err}")
        return 3

    print(f"[PASS] validated {len(blocks)} smoke block(s) from {args.smoke_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

