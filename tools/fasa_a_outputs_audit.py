#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _is_finite_list(value: Any, expected_len: int | None = None) -> bool:
    if not isinstance(value, list):
        return False
    if expected_len is not None and len(value) != expected_len:
        return False
    return all(_safe_float(item) is not None for item in value)


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
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 when gate_pass=false.")
    parser.add_argument("--max-print-bad", type=int, default=5, help="Max bad examples stored in summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows_total = 0
    rows_contract_ok = 0
    rows_region_len_mismatch = 0
    rows_embedding_len_mismatch = 0
    rows_nonfinite = 0
    bad_examples: list[dict[str, Any]] = []

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
                rows_nonfinite += 1
                row_errors.append(f"invalid json: {exc}")
                if len(bad_examples) < args.max_print_bad:
                    bad_examples.append({"line_no": line_no, "error": "; ".join(row_errors)})
                continue

            payload = _extract_payload(record)
            if payload is None:
                row_errors.append("a_outputs payload missing")
            else:
                for field in REQUIRED_FIELDS:
                    if field not in payload:
                        row_errors.append(f"missing field: {field}")

                risk_pred = _safe_float(payload.get("risk_pred"))
                trigger_logit = _safe_float(payload.get("trigger_logit"))
                delta_pred = _safe_float(payload.get("delta_pred"))
                if risk_pred is None:
                    row_errors.append("risk_pred non-finite")
                if trigger_logit is None:
                    row_errors.append("trigger_logit non-finite")
                if delta_pred is None or delta_pred < 0.0:
                    row_errors.append("delta_pred non-finite or negative")

                region_logits = payload.get("region_logits")
                if not isinstance(region_logits, list) or len(region_logits) != EXPECTED_REGION_LEN:
                    rows_region_len_mismatch += 1
                    row_errors.append(
                        f"region_logits len={len(region_logits) if isinstance(region_logits, list) else 'NA'} != {EXPECTED_REGION_LEN}"
                    )
                elif not _is_finite_list(region_logits):
                    rows_nonfinite += 1
                    row_errors.append("region_logits contains non-finite")

                dynamic_embedding = payload.get("dynamic_embedding")
                if not isinstance(dynamic_embedding, list) or len(dynamic_embedding) != EXPECTED_EMBED_LEN:
                    rows_embedding_len_mismatch += 1
                    row_errors.append(
                        f"dynamic_embedding len={len(dynamic_embedding) if isinstance(dynamic_embedding, list) else 'NA'} != {EXPECTED_EMBED_LEN}"
                    )
                elif not _is_finite_list(dynamic_embedding):
                    rows_nonfinite += 1
                    row_errors.append("dynamic_embedding contains non-finite")

                version = payload.get("version")
                source = payload.get("source")
                if not isinstance(version, str) or not version.strip():
                    row_errors.append("version missing/empty")
                if not isinstance(source, str) or not source.strip():
                    row_errors.append("source missing/empty")

            if row_errors:
                if len(bad_examples) < args.max_print_bad:
                    bad_examples.append({"line_no": line_no, "error": "; ".join(row_errors)})
            else:
                rows_contract_ok += 1

    core_gate_pass = (
        rows_total > 0
        and rows_contract_ok == rows_total
        and rows_region_len_mismatch == 0
        and rows_embedding_len_mismatch == 0
        and rows_nonfinite == 0
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
                fallback_counts = producer_summary.get("fallback_counts") if isinstance(producer_summary.get("fallback_counts"), dict) else {}
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

    summary = {
        "input_jsonl": str(path),
        "rows_total": rows_total,
        "rows_contract_ok": rows_contract_ok,
        "rows_contract_ok_ratio": float(rows_contract_ok / max(1, rows_total)),
        "rows_region_len_mismatch": rows_region_len_mismatch,
        "rows_embedding_len_mismatch": rows_embedding_len_mismatch,
        "rows_nonfinite": rows_nonfinite,
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
