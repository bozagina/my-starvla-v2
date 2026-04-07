#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_vector(value: Any, target_len: int) -> np.ndarray | None:
    if target_len <= 0:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = arr.reshape(-1)
    if arr.size >= target_len:
        return arr[:target_len]
    padded = np.zeros((target_len,), dtype=np.float32)
    padded[: arr.size] = arr
    return padded


def _prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = max(eps, min(1.0 - eps, prob))
    return float(math.log(clipped / (1.0 - clipped)))


def _derive_dynamic_embedding(
    record: dict,
    *,
    dim: int,
) -> list[float]:
    """Build a fixed-size embedding from available per-sample signals.

    This is a placeholder adapter until full VDPM feature extraction is wired.
    """
    features: list[float] = []
    remaining_chunk = record.get("remaining_chunk")
    if remaining_chunk is not None:
        try:
            arr = np.asarray(remaining_chunk, dtype=np.float32)
            if arr.ndim == 2 and arr.size > 0:
                flat = arr.reshape(-1)
                features.extend(
                    [
                        float(flat.mean()),
                        float(flat.std()),
                        float(flat.min()),
                        float(flat.max()),
                        float(np.abs(flat).mean()),
                        float(np.linalg.norm(flat) / max(1, flat.size)),
                    ]
                )
                t_norm = np.linalg.norm(arr, axis=-1)
                features.extend(
                    [
                        float(t_norm.mean()),
                        float(t_norm.std()),
                        float(t_norm.min()),
                        float(t_norm.max()),
                    ]
                )
        except Exception:
            pass

    pseudo = record.get("pseudo_labels")
    if isinstance(pseudo, dict):
        for key in ("risk_score", "trigger_label", "delta_action_norm"):
            value = _safe_float(pseudo.get(key))
            features.append(0.0 if value is None else float(value))

    if not features:
        features = [0.0]
    out = np.asarray(features, dtype=np.float32)
    if out.size < dim:
        out = np.pad(out, (0, dim - out.size), mode="constant")
    else:
        out = out[:dim]
    return out.tolist()


def _build_a_outputs_from_record(
    record: dict,
    *,
    embedding_dim: int,
    region_key: str,
) -> dict:
    pseudo = record.get("pseudo_labels")
    pseudo = pseudo if isinstance(pseudo, dict) else {}

    risk_pred = _safe_float(pseudo.get("risk_score"))
    if risk_pred is None:
        risk_pred = 0.0

    trigger_logit = _safe_float(pseudo.get("trigger_logit"))
    if trigger_logit is None:
        trigger_prob = _safe_float(pseudo.get("trigger_label"))
        trigger_logit = _prob_to_logit(trigger_prob) if trigger_prob is not None else 0.0

    delta_pred = _safe_float(pseudo.get("delta_pred"))
    if delta_pred is None:
        delta_pred = _safe_float(pseudo.get("delta_action_norm"))
    if delta_pred is None:
        delta_pred = 0.0

    chunk_len = 0
    remaining_chunk = record.get("remaining_chunk")
    if remaining_chunk is not None:
        try:
            arr = np.asarray(remaining_chunk)
            if arr.ndim == 2:
                chunk_len = int(arr.shape[0])
        except Exception:
            pass
    if chunk_len <= 0:
        correction_mask = pseudo.get("correction_mask")
        if correction_mask is not None:
            try:
                chunk_len = int(np.asarray(correction_mask).reshape(-1).size)
            except Exception:
                chunk_len = 0
    chunk_len = max(1, chunk_len)

    region_logits = None
    for key in (region_key, "affected_region_prior", "correction_mask"):
        region_logits = _coerce_vector(pseudo.get(key), target_len=chunk_len)
        if region_logits is not None:
            break
    if region_logits is None:
        region_logits = np.zeros((chunk_len,), dtype=np.float32)

    return {
        "version": "a_outputs_v1_placeholder",
        "source": "heuristic_from_pseudo_labels",
        "risk_pred": float(risk_pred),
        "trigger_logit": float(trigger_logit),
        "delta_pred": float(delta_pred),
        "region_logits": region_logits.astype(np.float32).tolist(),
        "dynamic_embedding": _derive_dynamic_embedding(record, dim=embedding_dim),
    }


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standalone A-module outputs (`a_outputs`) for correction dataset JSONL."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input correction dataset JSONL path.")
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="Output JSONL path. Default: <input>_with_a_outputs.jsonl",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Dimension of placeholder dynamic_embedding.",
    )
    parser.add_argument(
        "--region-key",
        default="region_logits",
        help="Preferred pseudo_labels key for region signal fallback.",
    )
    parser.add_argument(
        "--overwrite-existing-a-outputs",
        action="store_true",
        help="Overwrite record['a_outputs'] if already present.",
    )
    parser.add_argument(
        "--stats-json",
        default="",
        help="Optional path to save generation statistics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")

    output_path = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else input_path.with_name(f"{input_path.stem}_with_a_outputs.jsonl")
    )

    records = _load_jsonl(input_path)
    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_total": len(records),
        "rows_updated": 0,
        "rows_kept_existing": 0,
        "rows_missing_meta": 0,
        "rows_missing_pseudo_labels": 0,
    }

    updated_records = []
    for record in records:
        if not isinstance(record, dict):
            updated_records.append(record)
            continue
        if not isinstance(record.get("meta"), dict):
            stats["rows_missing_meta"] += 1

        pseudo = record.get("pseudo_labels")
        if not isinstance(pseudo, dict):
            stats["rows_missing_pseudo_labels"] += 1

        has_existing = isinstance(record.get("a_outputs"), dict)
        if has_existing and not args.overwrite_existing_a_outputs:
            stats["rows_kept_existing"] += 1
            updated_records.append(record)
            continue

        new_record = dict(record)
        new_record["a_outputs"] = _build_a_outputs_from_record(
            record,
            embedding_dim=max(1, int(args.embedding_dim)),
            region_key=str(args.region_key),
        )
        updated_records.append(new_record)
        stats["rows_updated"] += 1

    _write_jsonl(output_path, updated_records)
    print(
        f"[done] wrote {len(updated_records)} rows -> {output_path} "
        f"(updated={stats['rows_updated']}, kept_existing={stats['rows_kept_existing']})"
    )

    if args.stats_json:
        stats_path = Path(args.stats_json).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[done] wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()
