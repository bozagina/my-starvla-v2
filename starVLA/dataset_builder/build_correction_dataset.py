#!/usr/bin/env python3
"""Build a tiny correction pseudo-label dataset from StarVLA shared-builder samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.dataset_builder.pseudo_label_utils import compute_correction_pseudo_labels
from starVLA.dataset_builder.sample_schema import (
    CORRECTION_SCHEMA_VERSION,
    validate_correction_entry,
)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_2d_float32(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={array.shape}")
    return array


def _build_entry(sample: dict, index: int) -> dict:
    if "action_chunk" in sample:
        action_chunk = _to_2d_float32(sample["action_chunk"], "action_chunk")
    else:
        action_chunk = _to_2d_float32(sample["action"], "action")
    remaining_chunk = action_chunk[1:]

    pseudo_labels = compute_correction_pseudo_labels(action_chunk)
    meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else {}
    state = sample.get("state")

    entry = {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "lang": sample.get("lang", ""),
        "action_chunk": action_chunk.tolist(),
        "remaining_chunk": remaining_chunk.tolist(),
        "pseudo_labels": pseudo_labels,
        "meta": {
            "record_index": int(index),
            "source_schema_version": meta.get("schema_version", "legacy"),
            "dataset_name": meta.get("dataset_name"),
            "trajectory_id": meta.get("trajectory_id"),
            "sample_step": meta.get("sample_step"),
            "action_chunk_len": int(action_chunk.shape[0]),
            "action_dim": int(action_chunk.shape[1]),
        },
    }
    if state is not None:
        state_arr = _to_2d_float32(state, "state")
        entry["state"] = state_arr.tolist()
        entry["meta"]["state_dim"] = int(state_arr.shape[1])

    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to training yaml used for dataloader construction.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--run_id", type=str, default="p1_correction_dataset_builder_sanity")
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--data_mix", type=str, default=None)
    parser.add_argument("--video_backend", type=str, default=None)
    parser.add_argument("--include_state", type=str, default="true")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")

    cfg = OmegaConf.load(args.config_yaml)
    cfg.run_id = args.run_id
    cfg.output_dir = args.output_dir
    cfg.run_root_dir = str(Path(args.output_dir).parent)

    vla_cfg = cfg.datasets.vla_data
    vla_cfg.shared_builder_enabled = True
    vla_cfg.include_state = _to_bool(args.include_state, default=True)
    if args.data_root_dir:
        vla_cfg.data_root_dir = args.data_root_dir
    if args.data_mix:
        vla_cfg.data_mix = args.data_mix
    if args.video_backend:
        vla_cfg.video_backend = args.video_backend

    dataset_py = vla_cfg.get("dataset_py", "lerobot_datasets")
    dataloader = build_dataloader(cfg=cfg, dataset_py=dataset_py)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "correction_dataset.jsonl"
    summary_path = output_dir / "summary.json"

    total = 0
    trigger_pos = 0
    trigger_neg = 0
    risk_zero_trigger_one = 0

    with jsonl_path.open("w", encoding="utf-8") as f:
        for batch in dataloader:
            if not isinstance(batch, list):
                raise TypeError(f"Expected list batch from collate_fn, got {type(batch)}")
            for sample in batch:
                entry = _build_entry(sample, index=total)
                errors = validate_correction_entry(entry)
                if errors:
                    raise ValueError(f"schema validation failed at sample[{total}]: {errors}")

                pseudo = entry["pseudo_labels"]
                if pseudo["trigger_label"] == 1:
                    trigger_pos += 1
                else:
                    trigger_neg += 1
                if abs(float(pseudo["risk_score"])) <= 1e-8 and int(pseudo["trigger_label"]) == 1:
                    risk_zero_trigger_one += 1

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total += 1
                if total >= args.num_samples:
                    break
            if total >= args.num_samples:
                break

    summary = {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "num_samples": total,
        "output_jsonl": str(jsonl_path.resolve()),
        "trigger_distribution": {"0": trigger_neg, "1": trigger_pos},
        "risk_zero_trigger_one": risk_zero_trigger_one,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
