#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_REGION_LEN = 15
REQUIRED_EMBED_LEN = 16
FALLBACK_TYPES = ("risk", "trigger", "delta", "region", "embed")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_vector(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        item_f = _safe_float(item)
        if item_f is not None:
            out.append(item_f)
    return out


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _logit(prob: float) -> float:
    p = min(max(prob, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _reduce_delta(value: Any) -> tuple[float, bool]:
    scalar = _safe_float(value)
    if scalar is not None:
        return max(0.0, scalar), True
    if isinstance(value, list):
        vals = [_safe_float(item) for item in value]
        vals = [item for item in vals if item is not None]
        if vals:
            return max(0.0, max(vals)), True
    return 0.0, False


def _resample(values: list[float], target_len: int) -> list[float]:
    if target_len <= 0:
        return []
    if not values:
        return [0.0] * target_len
    if len(values) == target_len:
        return values[:]
    if len(values) == 1:
        return [values[0]] * target_len

    out: list[float] = []
    src_last = len(values) - 1
    dst_last = target_len - 1
    for idx in range(target_len):
        pos = (idx * src_last) / dst_last
        lo = int(math.floor(pos))
        hi = min(lo + 1, src_last)
        w = pos - lo
        val = (1.0 - w) * values[lo] + w * values[hi]
        out.append(val)
    return out


def _region_probs(record: dict[str, Any], pseudo: dict[str, Any]) -> tuple[list[float], str, bool, bool]:
    region = record.get("region_target_15")
    if isinstance(region, list) and len(region) == REQUIRED_REGION_LEN:
        vals = [_safe_float(item) for item in region]
        if all(item is not None for item in vals):
            return [_clip01(float(item)) for item in vals if item is not None], "region_target_15", False, True

    prior = _finite_vector(pseudo.get("affected_region_prior"))
    mask = _finite_vector(pseudo.get("correction_mask"))

    if prior and mask:
        rp = _resample(prior, REQUIRED_REGION_LEN)
        rm = _resample(mask, REQUIRED_REGION_LEN)
        return [_clip01(max(a, b)) for a, b in zip(rp, rm)], "max(prior,mask)", True, True
    if prior:
        return [_clip01(x) for x in _resample(prior, REQUIRED_REGION_LEN)], "affected_region_prior", True, True
    if mask:
        return [_clip01(x) for x in _resample(mask, REQUIRED_REGION_LEN)], "correction_mask", True, True
    return [0.0] * REQUIRED_REGION_LEN, "zeros", True, False


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(max(0.0, var))
    return mean, std, min(values), max(values)


def _dynamic_embedding(
    record: dict[str, Any], risk_pred: float, trigger_logit: float, delta_pred: float
) -> tuple[list[float], bool, bool]:
    state_t = _finite_vector(record.get("state_t"))
    future_state = _finite_vector(record.get("future_state"))
    horizon = float(_safe_int(record.get("horizon_steps")) or 0)

    has_signal = len(state_t) > 0 and len(future_state) > 0

    m = min(len(state_t), len(future_state))
    diff = [future_state[i] - state_t[i] for i in range(m)]
    abs_diff = [abs(x) for x in diff]

    diff_mean, diff_std, diff_min, diff_max = _stats(diff)
    abs_mean, _, _, abs_max = _stats(abs_diff)
    state_mean, state_std, _, state_max = _stats(state_t)
    fut_mean, fut_std, _, fut_max = _stats(future_state)

    l2 = math.sqrt(sum(x * x for x in diff)) if diff else 0.0

    feats = [
        risk_pred,
        trigger_logit,
        delta_pred,
        horizon,
        float(len(state_t)),
        float(len(future_state)),
        diff_mean,
        diff_std,
        diff_min,
        diff_max,
        abs_mean,
        abs_max,
        l2,
        state_mean,
        state_std,
        state_max,
        fut_mean,
        fut_std,
        fut_max,
    ]

    out: list[float] = []
    for v in feats:
        v_f = _safe_float(v)
        out.append(v_f if v_f is not None else 0.0)
    if len(out) < REQUIRED_EMBED_LEN:
        out.extend([0.0] * (REQUIRED_EMBED_LEN - len(out)))
    return out[:REQUIRED_EMBED_LEN], not has_signal, has_signal


def _build_payload(record: dict[str, Any], version: str, source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pseudo_raw = record.get("pseudo_labels")
    pseudo = pseudo_raw if isinstance(pseudo_raw, dict) else {}

    fallback_types: set[str] = set()
    missing_fields: list[str] = []

    if not isinstance(pseudo_raw, dict):
        missing_fields.append("pseudo_labels")

    risk_score = _safe_float(pseudo.get("risk_score"))
    if risk_score is None:
        fallback_types.add("risk")
        missing_fields.append("pseudo_labels.risk_score")
    risk_pred = _clip01(risk_score if risk_score is not None else 0.0)

    trigger_label = _safe_float(pseudo.get("trigger_label"))
    if trigger_label is None:
        fallback_types.add("trigger")
        missing_fields.append("pseudo_labels.trigger_label")
        trigger_logit = 0.0
    else:
        trigger_logit = 2.0 * (_clip01(trigger_label) - 0.5)

    delta_pred, has_delta = _reduce_delta(pseudo.get("delta_action_norm"))
    if not has_delta:
        fallback_types.add("delta")
        missing_fields.append("pseudo_labels.delta_action_norm")

    region_probs, region_source, region_fallback, has_region_signal = _region_probs(record, pseudo)
    if region_fallback:
        fallback_types.add("region")
    if not has_region_signal:
        missing_fields.append("region supervision")

    region_logits = [_logit(_clip01(p)) for p in region_probs]
    dynamic_embedding, embed_fallback, has_embed_signal = _dynamic_embedding(record, risk_pred, trigger_logit, delta_pred)
    if embed_fallback:
        fallback_types.add("embed")
    if not has_embed_signal:
        missing_fields.append("state_t/future_state")

    payload = {
        "risk_pred": float(risk_pred),
        "trigger_logit": float(trigger_logit),
        "delta_pred": float(delta_pred),
        "region_logits": [float(x) for x in region_logits],
        "dynamic_embedding": [float(x) for x in dynamic_embedding],
        "version": version,
        "source": source,
    }

    row_meta = {
        "fallback_types": sorted(fallback_types),
        "missing_fields": missing_fields,
        "region_source": region_source,
        "row_with_fallback_any": len(fallback_types) > 0,
        "row_with_missing_pseudo_labels": not isinstance(pseudo_raw, dict),
    }
    return payload, row_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline FASA a_outputs JSONL from Phase0 JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Input Phase0 JSONL path.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path with sidecar a_outputs payload.")
    parser.add_argument("--version", default="fasa_v1", help="Payload contract version.")
    parser.add_argument("--source", default="fasa/main", help="Payload source tag.")
    parser.add_argument("--sample-check-json", default=None, help="Optional sample check output JSON.")
    parser.add_argument("--summary-json", default=None, help="Optional path to write producer summary JSON.")
    parser.add_argument("--max-samples", type=int, default=5, help="Max sample rows written into sample check.")
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail with non-zero exit when key supervision fields are missing.",
    )
    parser.add_argument("--max-print-missing", type=int, default=5, help="Max missing examples stored in summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_total = 0
    rows_with_fallback_any = 0
    rows_with_missing_pseudo_labels = 0
    fallback_counts = {key: 0 for key in FALLBACK_TYPES}
    strict_missing_failed_rows = 0
    strict_missing_examples: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no}: expected dict JSON object")

            payload, row_meta = _build_payload(record, version=args.version, source=args.source)
            out_row = {
                "line_no": line_no,
                "meta": record.get("meta", {}),
                "horizon_steps": record.get("horizon_steps"),
                "a_outputs": payload,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            rows_total += 1

            if row_meta["row_with_missing_pseudo_labels"]:
                rows_with_missing_pseudo_labels += 1

            if row_meta["row_with_fallback_any"]:
                rows_with_fallback_any += 1
                for fallback_type in row_meta["fallback_types"]:
                    fallback_counts[fallback_type] += 1

            if args.strict_missing and row_meta["missing_fields"]:
                strict_missing_failed_rows += 1
                if len(strict_missing_examples) < max(1, args.max_print_missing):
                    strict_missing_examples.append(
                        {
                            "line_no": line_no,
                            "missing_fields": row_meta["missing_fields"],
                            "fallback_types": row_meta["fallback_types"],
                        }
                    )

            if len(sample_rows) < max(1, args.max_samples):
                sample_rows.append(
                    {
                        "line_no": line_no,
                        "meta": record.get("meta", {}),
                        "region_logits_len": len(payload["region_logits"]),
                        "dynamic_embedding_len": len(payload["dynamic_embedding"]),
                        "version": payload["version"],
                        "source": payload["source"],
                        "fallback_types": row_meta["fallback_types"],
                        "region_source": row_meta["region_source"],
                    }
                )

    rows_with_fallback_any_ratio = float(rows_with_fallback_any / max(1, rows_total))
    strict_missing_pass = strict_missing_failed_rows == 0

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "rows_total": rows_total,
        "rows_with_fallback_any": rows_with_fallback_any,
        "rows_with_fallback_any_ratio": rows_with_fallback_any_ratio,
        "rows_with_missing_pseudo_labels": rows_with_missing_pseudo_labels,
        "fallback_counts": fallback_counts,
        "strict_missing_enabled": bool(args.strict_missing),
        "strict_missing_failed_rows": strict_missing_failed_rows,
        "strict_missing_pass": strict_missing_pass,
        "strict_missing_examples": strict_missing_examples,
        "sample_rows": sample_rows,
        "contract": {
            "required_fields": [
                "risk_pred",
                "trigger_logit",
                "delta_pred",
                "region_logits",
                "dynamic_embedding",
                "version",
                "source",
            ],
            "region_logits_len": REQUIRED_REGION_LEN,
            "dynamic_embedding_len": REQUIRED_EMBED_LEN,
            "version": args.version,
            "source": args.source,
        },
    }

    if args.sample_check_json:
        sample_path = Path(args.sample_check_json).expanduser().resolve()
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict_missing and not strict_missing_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
