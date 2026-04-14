#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CANONICAL_REGION_LEN = 15


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _is_finite_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if _safe_float(item) is None:
            return False
    return True


def _reduce_delta_norm(value: Any) -> float | None:
    if _safe_float(value) is not None:
        return float(value)
    if not isinstance(value, list):
        return None
    cleaned = [_safe_float(item) for item in value]
    cleaned = [item for item in cleaned if item is not None]
    if not cleaned:
        return None
    return max(cleaned)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FASA Phase0 dataset integrity.")
    parser.add_argument("--input-jsonl", required=True, help="Phase0 JSONL path.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the audit summary JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 when the gate fails.")
    parser.add_argument("--max-print-bad", type=int, default=5, help="Max bad examples to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows_total = 0
    rows_future_state_backfilled_ok = 0
    rows_region_target_15_len_mismatch = 0
    rows_nonfinite = 0
    bad_examples: list[tuple[int, str]] = []

    trigger_labels: list[int] = []
    region_active_bins: list[int] = []
    zero_delta_total = 0
    zero_delta_triggers = 0
    d_H_values: list[float] = []
    has_outcome = False
    outcome_version: str = "none"
    rows_with_intermediate_states = 0
    rows_with_progress_t = 0
    rows_delta_action_norm_scalar = 0
    progress_t_values: list[float] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rows_total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            rows_nonfinite += 1
            bad_examples.append((line_no, f"invalid json: {exc}"))
            continue

        row_errors: list[str] = []
        meta = record.get("meta")
        if not isinstance(meta, dict):
            row_errors.append("meta missing/not dict")
            meta = {}

        dataset_name = meta.get("dataset_name")
        trajectory_id = _safe_int(meta.get("trajectory_id"))
        sample_step = _safe_int(meta.get("sample_step"))
        horizon_steps = _safe_int(record.get("horizon_steps"))
        future_state_step = _safe_int(record.get("future_state_step"))

        if not isinstance(dataset_name, str) or not dataset_name.strip():
            row_errors.append("meta.dataset_name missing/empty")
        if trajectory_id is None:
            row_errors.append("meta.trajectory_id missing/non-int")
        if sample_step is None:
            row_errors.append("meta.sample_step missing/non-int")
        if horizon_steps is None or horizon_steps <= 0:
            row_errors.append("horizon_steps missing/non-positive")
        if future_state_step is None:
            row_errors.append("future_state_step missing/non-int")

        state_t = record.get("state_t")
        future_state = record.get("future_state")
        if not _is_finite_list(state_t):
            row_errors.append("state_t missing/non-finite list")
        if not _is_finite_list(future_state):
            row_errors.append("future_state missing/non-finite list")

        region_target_15 = record.get("region_target_15")
        if not isinstance(region_target_15, list) or len(region_target_15) != CANONICAL_REGION_LEN:
            rows_region_target_15_len_mismatch += 1
            row_errors.append(
                f"region_target_15 len={len(region_target_15) if isinstance(region_target_15, list) else 'NA'} "
                f"!= {CANONICAL_REGION_LEN}"
            )
        elif not _is_finite_list(region_target_15):
            row_errors.append("region_target_15 contains non-finite values")

        pseudo = record.get("pseudo_labels")
        if not isinstance(pseudo, dict):
            row_errors.append("pseudo_labels missing/not dict")
            pseudo = {}
        else:
            if _safe_float(pseudo.get("risk_score")) is None:
                row_errors.append("pseudo_labels.risk_score missing/non-finite")
            trigger = _safe_float(pseudo.get("trigger_label"))
            if trigger is None:
                row_errors.append("pseudo_labels.trigger_label missing/non-finite")
            elif trigger not in (0.0, 1.0):
                row_errors.append(f"pseudo_labels.trigger_label={trigger} not in {{0,1}}")
            if _reduce_delta_norm(pseudo.get("delta_action_norm")) is None:
                row_errors.append("pseudo_labels.delta_action_norm missing/non-finite")

            trigger_val = int(trigger) if trigger is not None and trigger in (0.0, 1.0) else 0
            trigger_labels.append(trigger_val)

            delta_norms = pseudo.get("delta_action_norm")
            max_dn = _reduce_delta_norm(delta_norms)
            if max_dn is not None and max_dn < 1e-6:
                zero_delta_total += 1
                if trigger_val == 1:
                    zero_delta_triggers += 1

            outcome_v3 = pseudo.get("_outcome_v3")
            outcome_v2 = pseudo.get("_outcome_v2")
            if isinstance(outcome_v3, dict) and "d_H" in outcome_v3:
                has_outcome = True
                outcome_version = "v3"
                dh = _safe_float(outcome_v3["d_H"])
                if dh is not None:
                    d_H_values.append(dh)
            elif isinstance(outcome_v2, dict) and "d_H" in outcome_v2:
                has_outcome = True
                if outcome_version != "v3":
                    outcome_version = "v2"
                dh = _safe_float(outcome_v2["d_H"])
                if dh is not None:
                    d_H_values.append(dh)

        if isinstance(region_target_15, list) and len(region_target_15) == CANONICAL_REGION_LEN:
            active = sum(1 for v in region_target_15 if isinstance(v, (int, float)) and v > 0.5)
            region_active_bins.append(active)

        # v0.9.3 field checks
        int_states = record.get("intermediate_states")
        if isinstance(int_states, list) and len(int_states) > 0:
            if all(_is_finite_list(s) for s in int_states):
                rows_with_intermediate_states += 1
        pt = record.get("progress_t")
        if pt is not None:
            ptf = _safe_float(pt)
            if ptf is not None:
                rows_with_progress_t += 1
                progress_t_values.append(ptf)
        dan = pseudo.get("delta_action_norm") if isinstance(pseudo, dict) else None
        if isinstance(dan, (int, float)):
            rows_delta_action_norm_scalar += 1

        if (
            sample_step is not None
            and horizon_steps is not None
            and horizon_steps > 0
            and future_state_step is not None
            and future_state_step == sample_step + horizon_steps
            and future_state_step > sample_step
            and _is_finite_list(state_t)
            and _is_finite_list(future_state)
        ):
            rows_future_state_backfilled_ok += 1
        else:
            row_errors.append("future_state backfill check failed")

        if row_errors:
            rows_nonfinite += 1
            if len(bad_examples) < args.max_print_bad:
                bad_examples.append((line_no, "; ".join(row_errors)))

    n = rows_total
    trigger_positive_rate = float(sum(trigger_labels)) / n if n > 0 else 0.0
    mean_region_active = (
        float(sum(region_active_bins)) / len(region_active_bins) if region_active_bins else 0.0
    )
    zero_delta_false_trigger_rate = (
        float(zero_delta_triggers) / zero_delta_total if zero_delta_total > 0 else 0.0
    )

    ah1_pass = zero_delta_false_trigger_rate < 0.01
    ah2_pass = 0.01 <= trigger_positive_rate <= 0.30
    ah3_pass = mean_region_active < 5.0

    gate_pass = (
        rows_total > 0
        and rows_future_state_backfilled_ok == rows_total
        and rows_region_target_15_len_mismatch == 0
        and rows_nonfinite == 0
    )

    outcome_gate_pass = ah1_pass and ah2_pass and ah3_pass

    summary: dict[str, Any] = {
        "input_jsonl": str(path),
        "rows_total": rows_total,
        "rows_future_state_backfilled_ok": rows_future_state_backfilled_ok,
        "rows_future_state_backfilled_ok_ratio": float(rows_future_state_backfilled_ok / max(1, rows_total)),
        "rows_region_target_15_len_mismatch": rows_region_target_15_len_mismatch,
        "rows_nonfinite": rows_nonfinite,
        "gate_pass": gate_pass,
        "outcome_label_audit": {
            "has_outcome": has_outcome,
            "outcome_version": outcome_version,
            "AH1_zero_delta_false_trigger_rate": zero_delta_false_trigger_rate,
            "AH1_zero_delta_samples": zero_delta_total,
            "AH1_pass": ah1_pass,
            "AH2_trigger_positive_rate": trigger_positive_rate,
            "AH2_pass": ah2_pass,
            "AH3_mean_region_active_bins": mean_region_active,
            "AH3_pass": ah3_pass,
            "outcome_gate_pass": outcome_gate_pass,
        },
        "v093_fields": {
            "rows_with_intermediate_states": rows_with_intermediate_states,
            "rows_with_progress_t": rows_with_progress_t,
            "rows_delta_action_norm_scalar": rows_delta_action_norm_scalar,
            "progress_t_min": min(progress_t_values) if progress_t_values else None,
            "progress_t_max": max(progress_t_values) if progress_t_values else None,
        },
        "bad_examples": [
            {"line_no": line_no, "error": error}
            for line_no, error in bad_examples
        ],
    }
    if d_H_values:
        import numpy as _np
        d_arr = _np.array(d_H_values, dtype=_np.float64)
        summary["outcome_label_audit"]["d_H_count"] = len(d_H_values)
        summary["outcome_label_audit"]["d_H_mean"] = float(_np.mean(d_arr))
        summary["outcome_label_audit"]["d_H_P10"] = float(_np.quantile(d_arr, 0.10))
        summary["outcome_label_audit"]["d_H_P25"] = float(_np.quantile(d_arr, 0.25))
        summary["outcome_label_audit"]["d_H_P50"] = float(_np.quantile(d_arr, 0.50))
        summary["outcome_label_audit"]["d_H_P75"] = float(_np.quantile(d_arr, 0.75))
        summary["outcome_label_audit"]["d_H_P90"] = float(_np.quantile(d_arr, 0.90))
        summary["outcome_label_audit"]["d_H_P95"] = float(_np.quantile(d_arr, 0.95))
        summary["outcome_label_audit"]["d_H_max"] = float(_np.max(d_arr))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict and not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
