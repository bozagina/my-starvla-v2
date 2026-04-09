#!/usr/bin/env python3
"""
Validate StarVLA shared-builder sample contract from JSON payload or smoke log.

Supported payload shape (summary form):
{
  "shared_builder_enabled": true/false,
  "sample_keys": [...],
  "action_shape": [T, D],
  "action_chunk_shape": [T, D] or null,
  "state_shape": [...] or null,
  "meta": {...} or null
}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_2d(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(_is_int(x) for x in value):
        return None
    if value[0] <= 0 or value[1] <= 0:
        return None
    return [int(value[0]), int(value[1])]


def _load_payload_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("payload JSON must be an object")
    return obj


def _extract_json_blocks_from_log(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"SMOKE_RESULT_JSON_START\s*(\{.*?\})\s*SMOKE_RESULT_JSON_END", re.S)
    blocks: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except Exception:
            continue
        if isinstance(obj, dict):
            blocks.append(obj)
    return blocks


def _load_payload_from_log(path: Path, index: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    blocks = _extract_json_blocks_from_log(text)
    if not blocks:
        raise ValueError(f"no SMOKE_RESULT_JSON block found in {path}")
    if index < 0:
        index = len(blocks) + index
    if index < 0 or index >= len(blocks):
        raise IndexError(f"log index out of range: {index}, blocks={len(blocks)}")
    return blocks[index]


def _resolve_mode(payload: dict[str, Any], mode: str) -> str:
    if mode in {"shared", "legacy"}:
        return mode
    enabled = payload.get("shared_builder_enabled")
    if isinstance(enabled, bool):
        return "shared" if enabled else "legacy"
    sample_keys = payload.get("sample_keys")
    if isinstance(sample_keys, list) and "action_chunk" in sample_keys:
        return "shared"
    return "legacy"


def _validate_payload(
    payload: dict[str, Any],
    *,
    mode: str,
    expected_schema_version: str,
    expected_action_dim: int | None,
    expected_action_chunk_len: int | None,
    require_state: bool,
    strict_legacy_no_extra: bool,
) -> tuple[list[str], list[str], str]:
    errors: list[str] = []
    warnings: list[str] = []
    resolved_mode = _resolve_mode(payload, mode)

    sample_keys = payload.get("sample_keys")
    if not isinstance(sample_keys, list) or not all(isinstance(k, str) for k in sample_keys):
        errors.append("`sample_keys` must be a list[str].")
        sample_keys_set: set[str] = set()
    else:
        sample_keys_set = set(sample_keys)

    base_required = {"action", "image", "lang"}
    missing_base = sorted(base_required - sample_keys_set)
    if missing_base:
        errors.append(f"missing base keys: {missing_base}")

    if require_state and "state" not in sample_keys_set:
        errors.append("`state` is required by --require-state but missing in sample_keys.")

    action_shape = _shape_2d(payload.get("action_shape"))
    if action_shape is None:
        errors.append("`action_shape` must be [T, D] with positive integers.")
    else:
        if expected_action_chunk_len is not None and action_shape[0] != expected_action_chunk_len:
            errors.append(
                f"`action_shape[0]` mismatch: got {action_shape[0]}, expected {expected_action_chunk_len}."
            )
        if expected_action_dim is not None and action_shape[1] != expected_action_dim:
            errors.append(f"`action_shape[1]` mismatch: got {action_shape[1]}, expected {expected_action_dim}.")

    action_chunk_shape = _shape_2d(payload.get("action_chunk_shape"))
    meta = payload.get("meta")

    if resolved_mode == "shared":
        required_shared = {"obs", "action_chunk", "meta"}
        missing_shared = sorted(required_shared - sample_keys_set)
        if missing_shared:
            errors.append(f"shared mode missing keys: {missing_shared}")

        if action_chunk_shape is None:
            errors.append("shared mode requires `action_chunk_shape` = [T, D].")
        if action_shape is not None and action_chunk_shape is not None and action_chunk_shape != action_shape:
            errors.append(
                f"`action_chunk_shape` must equal `action_shape` in shared mode; got {action_chunk_shape} vs {action_shape}."
            )

        if not isinstance(meta, dict):
            errors.append("shared mode requires `meta` object.")
        else:
            schema_version = meta.get("schema_version")
            if schema_version != expected_schema_version:
                errors.append(
                    f"`meta.schema_version` mismatch: got {schema_version!r}, expected {expected_schema_version!r}."
                )

            for key in [
                "dataset_name",
                "trajectory_id",
                "sample_step",
                "action_keys",
                "state_keys",
                "video_keys",
                "action_chunk_len",
                "action_dim",
            ]:
                if key not in meta:
                    errors.append(f"`meta.{key}` is required in shared mode.")

            if action_chunk_shape is not None:
                if meta.get("action_chunk_len") != action_chunk_shape[0]:
                    errors.append(
                        f"`meta.action_chunk_len` mismatch: got {meta.get('action_chunk_len')}, expected {action_chunk_shape[0]}."
                    )
                if meta.get("action_dim") != action_chunk_shape[1]:
                    errors.append(
                        f"`meta.action_dim` mismatch: got {meta.get('action_dim')}, expected {action_chunk_shape[1]}."
                    )

            if expected_action_chunk_len is not None and meta.get("action_chunk_len") != expected_action_chunk_len:
                errors.append(
                    f"`meta.action_chunk_len` mismatch: got {meta.get('action_chunk_len')}, expected {expected_action_chunk_len}."
                )
            if expected_action_dim is not None and meta.get("action_dim") != expected_action_dim:
                errors.append(f"`meta.action_dim` mismatch: got {meta.get('action_dim')}, expected {expected_action_dim}.")

    else:
        if strict_legacy_no_extra:
            forbidden = sorted(sample_keys_set.intersection({"obs", "action_chunk", "meta"}))
            if forbidden:
                errors.append(f"legacy mode must not expose shared-only keys: {forbidden}")
            if payload.get("action_chunk_shape") is not None:
                errors.append("legacy mode expects `action_chunk_shape` to be null.")
            if payload.get("meta") is not None:
                errors.append("legacy mode expects `meta` to be null.")
        else:
            warnings.append("legacy mode extra keys are allowed by --allow-legacy-extra.")

    return errors, warnings, resolved_mode


def _demo_payload(kind: str) -> dict[str, Any]:
    if kind == "shared_ok":
        return {
            "shared_builder_enabled": True,
            "sample_keys": ["action", "action_chunk", "image", "lang", "meta", "obs", "state"],
            "action_shape": [16, 7],
            "action_chunk_shape": [16, 7],
            "state_shape": [1, 8],
            "meta": {
                "schema_version": "p1_shared_builder_v1",
                "dataset_name": "libero_goal_no_noops_1.0.0_lerobot",
                "trajectory_id": 0,
                "sample_step": 0,
                "action_keys": ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"],
                "state_keys": ["state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw", "state.pad", "state.gripper"],
                "video_keys": ["video.primary_image", "video.wrist_image"],
                "action_chunk_len": 16,
                "action_dim": 7,
            },
        }
    if kind == "legacy_ok":
        return {
            "shared_builder_enabled": False,
            "sample_keys": ["action", "image", "lang", "state"],
            "action_shape": [16, 7],
            "action_chunk_shape": None,
            "state_shape": [1, 8],
            "meta": None,
        }
    if kind == "shared_bad":
        return {
            "shared_builder_enabled": True,
            "sample_keys": ["action", "image", "lang", "state", "obs", "action_chunk", "meta"],
            "action_shape": [16, 7],
            "action_chunk_shape": [8, 7],
            "state_shape": [1, 8],
            "meta": {
                "schema_version": "broken_schema",
                "dataset_name": "bad",
                "trajectory_id": 1,
                "sample_step": 1,
                "action_keys": [],
                "state_keys": [],
                "video_keys": [],
                "action_chunk_len": 8,
                "action_dim": 7,
            },
        }
    raise ValueError(f"unsupported demo kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate shared-builder sample contract.")
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--payload-json", type=Path, help="Path to payload JSON object.")
    source_group.add_argument("--smoke-log", type=Path, help="Path to log containing SMOKE_RESULT_JSON blocks.")
    source_group.add_argument(
        "--demo",
        choices=["shared_ok", "legacy_ok", "shared_bad"],
        help="Print a built-in demo payload JSON and exit.",
    )

    parser.add_argument("--log-index", type=int, default=-1, help="When --smoke-log is used: pick JSON block index, default -1 (last).")
    parser.add_argument("--mode", choices=["auto", "shared", "legacy"], default="auto")
    parser.add_argument("--expected-schema-version", default="p1_shared_builder_v1")
    parser.add_argument("--expected-action-dim", type=int, default=None)
    parser.add_argument("--expected-action-chunk-len", type=int, default=None)
    parser.add_argument("--require-state", action="store_true")
    parser.add_argument("--allow-legacy-extra", action="store_true", help="Allow legacy payload to carry shared-only keys.")
    args = parser.parse_args()

    if args.demo:
        print(json.dumps(_demo_payload(args.demo), ensure_ascii=False, indent=2))
        return 0

    if args.payload_json is None and args.smoke_log is None:
        parser.error("one source is required: --payload-json, --smoke-log, or --demo")

    if args.payload_json is not None:
        payload = _load_payload_json(args.payload_json)
        source = str(args.payload_json)
    else:
        payload = _load_payload_from_log(args.smoke_log, args.log_index)
        source = f"{args.smoke_log}#index={args.log_index}"

    errors, warnings, resolved_mode = _validate_payload(
        payload,
        mode=args.mode,
        expected_schema_version=args.expected_schema_version,
        expected_action_dim=args.expected_action_dim,
        expected_action_chunk_len=args.expected_action_chunk_len,
        require_state=args.require_state,
        strict_legacy_no_extra=not args.allow_legacy_extra,
    )

    print("=== Shared Builder Schema Validator ===")
    print(f"source={source}")
    print(f"mode={resolved_mode}")
    print(f"sample_keys={payload.get('sample_keys')}")
    print(f"action_shape={payload.get('action_shape')}")
    print(f"action_chunk_shape={payload.get('action_chunk_shape')}")
    meta = payload.get("meta")
    if isinstance(meta, dict):
        print(
            "meta_summary="
            f"schema={meta.get('schema_version')}, "
            f"chunk_len={meta.get('action_chunk_len')}, "
            f"action_dim={meta.get('action_dim')}"
        )
    else:
        print("meta_summary=None")

    if warnings:
        print("warnings:")
        for idx, item in enumerate(warnings, start=1):
            print(f"  {idx}. {item}")

    if errors:
        print("errors:")
        for idx, item in enumerate(errors, start=1):
            print(f"  {idx}. {item}")
        print("VALIDATION_RESULT=FAIL")
        return 2

    print("VALIDATION_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
