#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _reduce_delta_norm(value: Any) -> float | None:
    if _is_finite_number(value):
        return float(value)
    if not isinstance(value, list):
        return None
    cleaned = [float(v) for v in value if _is_finite_number(v)]
    if not cleaned:
        return None
    return max(cleaned)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _check_vector(name: str, value: Any) -> str | None:
    if not isinstance(value, list):
        return f"{name} is not list"
    for i, item in enumerate(value):
        if not _is_finite_number(item):
            return f"{name}[{i}] is non-finite: {item!r}"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit correction JSONL a_outputs quality.")
    parser.add_argument("--input-jsonl", required=True, help="Path to correction JSONL with a_outputs.")
    parser.add_argument("--max-print-bad", type=int, default=5, help="Max invalid rows to print.")
    parser.add_argument(
        "--expected-region-len",
        type=int,
        default=None,
        help="Optional expected length for a_outputs.region_logits.",
    )
    parser.add_argument(
        "--expected-embedding-len",
        type=int,
        default=None,
        help="Optional expected length for a_outputs.dynamic_embedding.",
    )
    parser.add_argument(
        "--require-a-outputs",
        action="store_true",
        help="Treat rows missing a_outputs as failure candidates.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 when quality gate fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows_total = 0
    rows_missing_a_outputs = 0
    rows_with_nonfinite = 0
    rows_trigger0_but_risk_positive = 0
    rows_trigger_sign_mismatch = 0
    rows_region_len_mismatch = 0
    rows_embedding_len_mismatch = 0

    bad_examples: list[tuple[int, str]] = []
    a_version_counter: Counter[str] = Counter()
    a_source_counter: Counter[str] = Counter()

    risk_preds: list[float] = []
    delta_preds: list[float] = []
    trigger_logits: list[float] = []
    region_lengths: list[float] = []
    embedding_lengths: list[float] = []
    risk_abs_diff: list[float] = []
    delta_abs_diff: list[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                rows_with_nonfinite += 1
                bad_examples.append((line_no, f"invalid json: {exc}"))
                continue

            pseudo = record.get("pseudo_labels")
            pseudo = pseudo if isinstance(pseudo, dict) else {}
            trigger_label = pseudo.get("trigger_label")
            pseudo_risk = _safe_float(pseudo.get("risk_score"))
            if trigger_label == 0 and pseudo_risk is not None and pseudo_risk > 1e-8:
                rows_trigger0_but_risk_positive += 1

            a_outputs = record.get("a_outputs")
            if not isinstance(a_outputs, dict):
                rows_missing_a_outputs += 1
                if args.require_a_outputs:
                    bad_examples.append((line_no, "missing a_outputs dict"))
                continue

            a_version = str(a_outputs.get("version", ""))
            a_source = str(a_outputs.get("source", ""))
            a_version_counter[a_version] += 1
            a_source_counter[a_source] += 1

            row_errors: list[str] = []

            risk_pred = _safe_float(a_outputs.get("risk_pred"))
            trigger_logit = _safe_float(a_outputs.get("trigger_logit"))
            delta_pred = _safe_float(a_outputs.get("delta_pred"))
            if risk_pred is None:
                row_errors.append("risk_pred missing/non-finite")
            else:
                risk_preds.append(risk_pred)
            if trigger_logit is None:
                row_errors.append("trigger_logit missing/non-finite")
            else:
                trigger_logits.append(trigger_logit)
            if delta_pred is None:
                row_errors.append("delta_pred missing/non-finite")
            else:
                delta_preds.append(delta_pred)

            region_logits = a_outputs.get("region_logits")
            region_err = _check_vector("region_logits", region_logits)
            if region_err is not None:
                row_errors.append(region_err)
            else:
                region_lengths.append(float(len(region_logits)))
                if args.expected_region_len is not None and len(region_logits) != int(args.expected_region_len):
                    rows_region_len_mismatch += 1
                    row_errors.append(
                        f"region_logits len={len(region_logits)} != expected_region_len={int(args.expected_region_len)}"
                    )

            dynamic_embedding = a_outputs.get("dynamic_embedding")
            embed_err = _check_vector("dynamic_embedding", dynamic_embedding)
            if embed_err is not None:
                row_errors.append(embed_err)
            else:
                embedding_lengths.append(float(len(dynamic_embedding)))
                if args.expected_embedding_len is not None and len(dynamic_embedding) != int(args.expected_embedding_len):
                    rows_embedding_len_mismatch += 1
                    row_errors.append(
                        "dynamic_embedding len="
                        f"{len(dynamic_embedding)} != expected_embedding_len={int(args.expected_embedding_len)}"
                    )

            if trigger_logit is not None and trigger_label in (0, 1):
                pred_label = 1 if trigger_logit > 0.0 else 0
                if pred_label != int(trigger_label):
                    rows_trigger_sign_mismatch += 1

            if pseudo_risk is not None and risk_pred is not None:
                risk_abs_diff.append(abs(risk_pred - pseudo_risk))

            pseudo_delta = _reduce_delta_norm(pseudo.get("delta_action_norm"))
            if pseudo_delta is not None and delta_pred is not None:
                delta_abs_diff.append(abs(delta_pred - pseudo_delta))

            if row_errors:
                rows_with_nonfinite += 1
                bad_examples.append((line_no, "; ".join(row_errors)))

    gate_pass = rows_with_nonfinite == 0 and (not args.require_a_outputs or rows_missing_a_outputs == 0)
    summary = {
        "input_jsonl": str(path),
        "rows_total": rows_total,
        "rows_missing_a_outputs": rows_missing_a_outputs,
        "rows_with_nonfinite_or_invalid": rows_with_nonfinite,
        "rows_trigger0_but_risk_positive": rows_trigger0_but_risk_positive,
        "rows_trigger_sign_mismatch": rows_trigger_sign_mismatch,
        "rows_region_len_mismatch": rows_region_len_mismatch,
        "rows_embedding_len_mismatch": rows_embedding_len_mismatch,
        "a_outputs_versions": dict(a_version_counter),
        "a_outputs_sources": dict(a_source_counter),
        "risk_pred_stats": _summary(risk_preds),
        "delta_pred_stats": _summary(delta_preds),
        "trigger_logit_stats": _summary(trigger_logits),
        "region_length_stats": _summary(region_lengths),
        "embedding_length_stats": _summary(embedding_lengths),
        "risk_abs_diff_vs_pseudo_stats": _summary(risk_abs_diff),
        "delta_abs_diff_vs_pseudo_stats": _summary(delta_abs_diff),
        "gate_pass": gate_pass,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if bad_examples:
        limit = max(0, int(args.max_print_bad))
        for line_no, reason in bad_examples[:limit]:
            print(f"[bad-row] line={line_no} reason={reason}")

    if args.strict and not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
