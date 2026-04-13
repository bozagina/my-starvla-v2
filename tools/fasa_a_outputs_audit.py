#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = [
    "risk_pred",
    "trigger_logit",
    "delta_pred",
    "region_logits",
    "dynamic_embedding",
    "version",
    "source",
]
EXPECTED_REGION_LEN = 15
EXPECTED_EMBED_LEN = 16


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


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _check_vector(name: str, value: Any) -> str | None:
    if not isinstance(value, list):
        return f"{name} is not list"
    for i, item in enumerate(value):
        if not _is_finite_number(item):
            return f"{name}[{i}] is non-finite: {item!r}"
    return None


def _extract_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    payload = record.get("a_outputs")
    if isinstance(payload, dict):
        return payload
    if all(k in record for k in REQUIRED_FIELDS):
        return record
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict audit for offline FASA a_outputs JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL path containing a_outputs payloads.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the audit JSON summary.")
    parser.add_argument("--producer-summary-json", default=None, help="Optional producer summary JSON path.")
    parser.add_argument("--max-fallback-ratio", type=float, default=None, help="If set, fail fallback gate when ratio exceeds this value.")
    parser.add_argument("--max-print-bad", type=int, default=5, help="Max bad examples stored or printed.")
    parser.add_argument("--expected-region-len", type=int, default=EXPECTED_REGION_LEN)
    parser.add_argument("--expected-embedding-len", type=int, default=EXPECTED_EMBED_LEN)
    parser.add_argument("--require-a-outputs", action="store_true", help="Treat rows missing a_outputs as failures.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 when gate_pass=false.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows_total = 0
    rows_contract_ok = 0
    rows_missing_a_outputs = 0
    rows_with_nonfinite_or_invalid = 0
    rows_trigger0_but_risk_positive = 0
    rows_trigger_sign_mismatch = 0
    rows_region_len_mismatch = 0
    rows_embedding_len_mismatch = 0
    rows_missing_version = 0
    rows_missing_source = 0
    rows_missing_frozen_keys = 0
    rows_with_all_frozen_fields = 0

    bad_examples: list[dict[str, Any]] = []
    a_version_counter: Counter[str] = Counter()
    a_source_counter: Counter[str] = Counter()
    risk_preds: list[float] = []
    delta_preds: list[float] = []
    trigger_logits: list[float] = []
    region_lengths: list[float] = []
    embedding_lengths: list[float] = []
    risk_abs_diff: list[float] = []
    delta_abs_diff: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            rows_total += 1
            row_errors: list[str] = []

            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                rows_with_nonfinite_or_invalid += 1
                if len(bad_examples) < args.max_print_bad:
                    bad_examples.append({"line_no": line_no, "error": f"invalid json: {exc}"})
                continue

            pseudo = record.get("pseudo_labels")
            pseudo = pseudo if isinstance(pseudo, dict) else {}
            trigger_label = pseudo.get("trigger_label")
            pseudo_risk = _safe_float(pseudo.get("risk_score"))
            if trigger_label == 0 and pseudo_risk is not None and pseudo_risk > 1e-8:
                rows_trigger0_but_risk_positive += 1

            payload = _extract_payload(record)
            if payload is None:
                rows_missing_a_outputs += 1
                row_errors.append("a_outputs payload missing")
                if args.require_a_outputs:
                    rows_with_nonfinite_or_invalid += 1
            else:
                a_version = str(payload.get("version", ""))
                a_source = str(payload.get("source", ""))
                a_version_counter[a_version] += 1
                a_source_counter[a_source] += 1

                frozen_keys_found = 0
                for field in REQUIRED_FIELDS:
                    if field not in payload:
                        row_errors.append(f"missing field: {field}")

                risk_pred = _safe_float(payload.get("risk_pred"))
                trigger_logit = _safe_float(payload.get("trigger_logit"))
                delta_pred = _safe_float(payload.get("delta_pred"))
                if risk_pred is None:
                    row_errors.append("risk_pred non-finite")
                else:
                    risk_preds.append(risk_pred)
                    frozen_keys_found += 1
                if trigger_logit is None:
                    row_errors.append("trigger_logit non-finite")
                else:
                    trigger_logits.append(trigger_logit)
                    frozen_keys_found += 1
                if delta_pred is None or delta_pred < 0.0:
                    row_errors.append("delta_pred non-finite or negative")
                else:
                    delta_preds.append(delta_pred)
                    frozen_keys_found += 1

                region_logits = payload.get("region_logits")
                region_err = _check_vector("region_logits", region_logits)
                if region_err is not None:
                    row_errors.append(region_err)
                else:
                    region_lengths.append(float(len(region_logits)))
                    frozen_keys_found += 1
                    if len(region_logits) != int(args.expected_region_len):
                        rows_region_len_mismatch += 1
                        row_errors.append(
                            f"region_logits len={len(region_logits)} != expected_region_len={int(args.expected_region_len)}"
                        )

                dynamic_embedding = payload.get("dynamic_embedding")
                embed_err = _check_vector("dynamic_embedding", dynamic_embedding)
                if embed_err is not None:
                    row_errors.append(embed_err)
                else:
                    embedding_lengths.append(float(len(dynamic_embedding)))
                    frozen_keys_found += 1
                    if len(dynamic_embedding) != int(args.expected_embedding_len):
                        rows_embedding_len_mismatch += 1
                        row_errors.append(
                            f"dynamic_embedding len={len(dynamic_embedding)} != expected_embedding_len={int(args.expected_embedding_len)}"
                        )

                version = _nonempty_string(payload.get("version"))
                if version is None:
                    rows_missing_version += 1
                    row_errors.append("version missing/empty")
                else:
                    frozen_keys_found += 1

                source = _nonempty_string(payload.get("source"))
                if source is None:
                    rows_missing_source += 1
                    row_errors.append("source missing/empty")
                else:
                    frozen_keys_found += 1

                if frozen_keys_found == 7:
                    rows_with_all_frozen_fields += 1
                else:
                    rows_missing_frozen_keys += 1

                if trigger_logit is not None and trigger_label in (0, 1):
                    pred_label = 1 if trigger_logit > 0.0 else 0
                    if pred_label != int(trigger_label):
                        rows_trigger_sign_mismatch += 1

                if pseudo_risk is not None and risk_pred is not None:
                    risk_abs_diff.append(abs(risk_pred - pseudo_risk))
                pseudo_delta = _reduce_delta_norm(pseudo.get("delta_action_norm"))
                if pseudo_delta is not None and delta_pred is not None and delta_pred >= 0.0:
                    delta_abs_diff.append(abs(delta_pred - pseudo_delta))

            if row_errors:
                if payload is not None:
                    rows_with_nonfinite_or_invalid += 1
                if len(bad_examples) < args.max_print_bad:
                    bad_examples.append({"line_no": line_no, "error": "; ".join(row_errors)})
            elif payload is not None:
                rows_contract_ok += 1

    core_gate_pass = (
        rows_total > 0
        and rows_contract_ok == rows_total
        and rows_region_len_mismatch == 0
        and rows_embedding_len_mismatch == 0
        and rows_with_nonfinite_or_invalid == 0
        and (not args.require_a_outputs or rows_missing_a_outputs == 0)
    )

    producer_summary: dict[str, Any] | None = None
    fallback_summary_present = False
    rows_with_fallback_any = None
    rows_with_fallback_any_ratio = None
    fallback_counts: dict[str, Any] = {}
    fallback_gate_pass = True
    fallback_gate_reason = "not_checked"

    if args.producer_summary_json:
        summary_path = Path(args.producer_summary_json).expanduser().resolve()
        if not summary_path.exists():
            fallback_gate_pass = False
            fallback_gate_reason = f"producer_summary_missing:{summary_path}"
        else:
            producer_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(producer_summary, dict):
                fallback_summary_present = True
                rows_with_fallback_any = producer_summary.get("rows_with_fallback_any")
                rows_with_fallback_any_ratio = producer_summary.get("rows_with_fallback_any_ratio")
                fallback_counts = (
                    producer_summary.get("fallback_counts")
                    if isinstance(producer_summary.get("fallback_counts"), dict)
                    else {}
                )
                ratio_f = _safe_float(rows_with_fallback_any_ratio)
                if ratio_f is None and _safe_float(rows_with_fallback_any) is not None and _safe_float(producer_summary.get("rows_total")) is not None:
                    n = float(rows_with_fallback_any)
                    d = max(1.0, float(producer_summary.get("rows_total")))
                    ratio_f = n / d
                    rows_with_fallback_any_ratio = ratio_f
                if args.max_fallback_ratio is not None:
                    if ratio_f is None:
                        fallback_gate_pass = False
                        fallback_gate_reason = "fallback_ratio_missing"
                    elif ratio_f > args.max_fallback_ratio:
                        fallback_gate_pass = False
                        fallback_gate_reason = f"fallback_ratio_exceeds:{ratio_f}>{args.max_fallback_ratio}"
                    else:
                        fallback_gate_reason = f"fallback_ratio_ok:{ratio_f}<={args.max_fallback_ratio}"
                else:
                    fallback_gate_reason = "fallback_threshold_not_set"

    gate_pass = core_gate_pass and fallback_gate_pass
    region_ratio_denom = max(1, rows_total - rows_missing_a_outputs)
    embedding_ratio_denom = max(1, rows_total - rows_missing_a_outputs)
    summary = {
        "input_jsonl": str(path),
        "rows_total": rows_total,
        "rows_contract_ok": rows_contract_ok,
        "rows_contract_ok_ratio": float(rows_contract_ok / max(1, rows_total)),
        "rows_missing_a_outputs": rows_missing_a_outputs,
        "rows_with_nonfinite_or_invalid": rows_with_nonfinite_or_invalid,
        "rows_trigger0_but_risk_positive": rows_trigger0_but_risk_positive,
        "rows_trigger_sign_mismatch": rows_trigger_sign_mismatch,
        "rows_region_len_mismatch": rows_region_len_mismatch,
        "rows_embedding_len_mismatch": rows_embedding_len_mismatch,
        "rows_region_len_mismatch_ratio": float(rows_region_len_mismatch / region_ratio_denom),
        "rows_embedding_len_mismatch_ratio": float(rows_embedding_len_mismatch / embedding_ratio_denom),
        "rows_missing_version": rows_missing_version,
        "rows_missing_source": rows_missing_source,
        "rows_missing_frozen_keys": rows_missing_frozen_keys,
        "rows_with_all_frozen_fields": rows_with_all_frozen_fields,
        "rows_with_all_frozen_fields_ratio": float(rows_with_all_frozen_fields / max(1, rows_total)),
        "a_outputs_versions": dict(a_version_counter),
        "a_outputs_sources": dict(a_source_counter),
        "risk_pred_stats": _summary(risk_preds),
        "delta_pred_stats": _summary(delta_preds),
        "trigger_logit_stats": _summary(trigger_logits),
        "region_length_stats": _summary(region_lengths),
        "embedding_length_stats": _summary(embedding_lengths),
        "risk_abs_diff_vs_pseudo_stats": _summary(risk_abs_diff),
        "delta_abs_diff_vs_pseudo_stats": _summary(delta_abs_diff),
        "core_gate_pass": core_gate_pass,
        "fallback_summary_present": fallback_summary_present,
        "rows_with_fallback_any": rows_with_fallback_any,
        "rows_with_fallback_any_ratio": rows_with_fallback_any_ratio,
        "fallback_counts": fallback_counts,
        "max_fallback_ratio": args.max_fallback_ratio,
        "fallback_gate_pass": fallback_gate_pass,
        "fallback_gate_reason": fallback_gate_reason,
        "gate_pass": gate_pass,
        "bad_examples": bad_examples,
    }

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict and not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
