#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PHASE0_SCHEMA_VERSION = "p4_1_fasa_phase0_v1"
CANONICAL_REGION_LEN = 15

DEFAULT_TAU_S = 0.15
DEFAULT_TAU_FAIL = 0.7
DEFAULT_ALPHA_D = 0.35
DEFAULT_BETA_DELTA = 0.5
DEFAULT_GAMMA_REGION = 0.3

GROUP_WEIGHTS_7DOF = {
    "position": (slice(0, 3), 1.0),
    "rotation": (slice(3, 6), 0.5),
    "gripper": (slice(6, 7), 2.0),
}


def _build_weight_vector(state_dim: int) -> np.ndarray:
    """Build per-dimension weight vector W for state deviation."""
    if state_dim >= 7:
        w = np.ones(state_dim, dtype=np.float64)
        for _name, (slc, weight) in GROUP_WEIGHTS_7DOF.items():
            w[slc] = weight
        return w
    return np.ones(state_dim, dtype=np.float64)


def _raw_weighted_deviation(state_t: np.ndarray, future_state: np.ndarray) -> float:
    """Compute raw (unnormalized) weighted L2 state deviation."""
    state_dim = state_t.shape[0]
    w = _build_weight_vector(state_dim)
    diff = future_state.astype(np.float64) - state_t.astype(np.float64)
    weighted = w[:len(diff)] * diff[:len(w)]
    return float(np.sqrt(np.sum(weighted ** 2)))


def _calibrate_tau_s(raw_deviations: list[float], user_tau_s: float) -> tuple[float, dict[str, Any]]:
    """Calibrate τ_s from data distribution per spec §3.2.

    Returns (calibrated_tau_s, calibration_info).
    """
    if not raw_deviations:
        return user_tau_s, {"method": "default", "reason": "no_samples"}

    arr = np.array(raw_deviations, dtype=np.float64)
    p90 = float(np.quantile(arr, 0.90))

    info: dict[str, Any] = {
        "raw_deviation_P90": p90,
        "raw_deviation_mean": float(np.mean(arr)),
        "raw_deviation_max": float(np.max(arr)),
        "raw_deviation_min": float(np.min(arr)),
        "user_tau_s": user_tau_s,
        "n_samples": len(raw_deviations),
    }

    if p90 < 0.05:
        calibrated = 0.05
        info["method"] = "fallback_low"
        info["reason"] = f"P90={p90:.6f} < 0.05"
        info["warning"] = "data has near-zero state deviations"
    elif p90 > 0.5:
        calibrated = p90
        info["method"] = "fallback_high"
        info["reason"] = f"P90={p90:.6f} > 0.5"
        info["warning"] = "data has unusually large state deviations"
    else:
        calibrated = p90
        info["method"] = "p90_calibrated"
        info["reason"] = f"P90={p90:.6f} in normal range [0.05, 0.5]"

    info["calibrated_tau_s"] = calibrated
    return calibrated, info


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


def _flatten_vector(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(arr).all():
        return None
    return arr.astype(np.float32)


def _to_2d_float32(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(arr).all():
        return None
    return arr.astype(np.float32)


def _resample_vector(value: Any, target_len: int) -> np.ndarray | None:
    if target_len <= 0:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == target_len:
        return arr.astype(np.float32)
    if arr.size == 1:
        return np.full((target_len,), float(arr[0]), dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=arr.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst, src, arr).astype(np.float32)


def _compute_pseudo_labels(action_chunk: np.ndarray) -> dict[str, Any]:
    """Legacy delta-only pseudo labels (kept for --legacy-labels fallback)."""
    if action_chunk.ndim != 2 or action_chunk.shape[0] <= 0 or action_chunk.shape[1] <= 0:
        raise ValueError(f"action_chunk must be rank-2 with positive dims, got {action_chunk.shape}")

    if action_chunk.shape[0] == 1:
        delta_norm = np.zeros((0,), dtype=np.float32)
    else:
        delta_norm = np.linalg.norm(np.diff(action_chunk, axis=0), axis=1).astype(np.float32)

    if delta_norm.size == 0:
        correction_mask = np.zeros((0,), dtype=np.int32)
        risk_score = 0.0
        affected_region_prior = np.zeros((0,), dtype=np.float32)
    else:
        threshold = float(np.quantile(delta_norm, 0.75))
        correction_mask = (delta_norm >= threshold).astype(np.int32)
        risk_score = float(np.max(delta_norm))
        denom = float(np.sum(delta_norm))
        if denom > 0.0:
            affected_region_prior = (delta_norm / denom).astype(np.float32)
        else:
            affected_region_prior = np.full_like(delta_norm, 1.0 / len(delta_norm), dtype=np.float32)

    trigger_label = int(correction_mask.max()) if correction_mask.size > 0 else 0
    return {
        "trigger_label": trigger_label,
        "risk_score": risk_score,
        "affected_region_prior": affected_region_prior.tolist(),
        "correction_mask": correction_mask.tolist(),
        "delta_action_norm": delta_norm.tolist(),
    }


def _compute_pseudo_labels_v2(
    action_chunk: np.ndarray,
    state_t: np.ndarray,
    future_state: np.ndarray,
    *,
    tau_s: float = DEFAULT_TAU_S,
    tau_fail: float = DEFAULT_TAU_FAIL,
    alpha_d: float = DEFAULT_ALPHA_D,
    beta_delta: float = DEFAULT_BETA_DELTA,
    gamma_region: float = DEFAULT_GAMMA_REGION,
) -> dict[str, Any]:
    """Outcome-based pseudo labels per spec §3.4 (v0.9)."""
    if action_chunk.ndim != 2 or action_chunk.shape[0] <= 0 or action_chunk.shape[1] <= 0:
        raise ValueError(f"action_chunk must be rank-2 with positive dims, got {action_chunk.shape}")

    state_dim = state_t.shape[0]
    w = _build_weight_vector(state_dim)

    # §3.2: weighted state deviation d_H
    diff_state = (future_state.astype(np.float64) - state_t.astype(np.float64))
    weighted_diff = w[:len(diff_state)] * diff_state[:len(w)]
    raw_deviation = float(np.sqrt(np.sum(weighted_diff ** 2)))

    effective_tau_s = tau_s
    if effective_tau_s <= 0:
        effective_tau_s = DEFAULT_TAU_S
    d_H = float(np.clip(raw_deviation / effective_tau_s, 0.0, 1.0))

    # §3.3: failure detector f_H
    f_H = 1 if d_H > tau_fail else 0

    # §3.4 risk_score
    risk_score = float(np.clip(max(f_H, d_H), 0.0, 1.0))

    # §3.4 trigger_label
    trigger_label = 1 if (f_H == 1 or d_H > alpha_d) else 0

    # delta_norm (original action diff, preserved for compatibility)
    if action_chunk.shape[0] == 1:
        delta_norm = np.zeros((0,), dtype=np.float32)
    else:
        delta_norm = np.linalg.norm(np.diff(action_chunk, axis=0), axis=1).astype(np.float32)

    # §3.4 delta_action_norm: max(max_j ||Δa_j||, β·d_H)
    max_delta = float(delta_norm.max()) if delta_norm.size > 0 else 0.0
    delta_action_norm_scalar = max(max_delta, beta_delta * d_H)

    # §3.4 correction_mask: original Q75 OR f_H==1
    if delta_norm.size == 0:
        correction_mask = np.zeros((0,), dtype=np.int32)
    else:
        threshold = float(np.quantile(delta_norm, 0.75))
        q75_mask = delta_norm >= threshold
        if f_H == 1:
            correction_mask = np.ones_like(delta_norm, dtype=np.int32)
        else:
            correction_mask = q75_mask.astype(np.int32)

    # §3.4 affected_region_prior with outcome signal in second half
    if delta_norm.size == 0:
        affected_region_prior = np.zeros((0,), dtype=np.float32)
    else:
        denom = float(np.sum(delta_norm))
        if denom > 0.0:
            delta_norm_prior = (delta_norm / denom).astype(np.float64)
        else:
            delta_norm_prior = np.full(len(delta_norm), 1.0 / len(delta_norm), dtype=np.float64)
        half = len(delta_norm_prior) // 2
        region_prior = delta_norm_prior.copy()
        region_prior[half:] = np.clip(region_prior[half:] + gamma_region * d_H, 0.0, 1.0)
        region_prior[:half] = np.clip(region_prior[:half], 0.0, 1.0)
        affected_region_prior = region_prior.astype(np.float32)

    return {
        "trigger_label": trigger_label,
        "risk_score": risk_score,
        "affected_region_prior": affected_region_prior.tolist(),
        "correction_mask": correction_mask.tolist(),
        "delta_action_norm": delta_norm.tolist(),
        "_outcome_v2": {
            "d_H": d_H,
            "f_H": f_H,
            "raw_deviation": raw_deviation,
            "tau_s_effective": effective_tau_s,
            "tau_fail": tau_fail,
            "alpha_d": alpha_d,
            "beta_delta": beta_delta,
            "gamma_region": gamma_region,
            "delta_action_norm_scalar": delta_action_norm_scalar,
            "state_dim": state_dim,
            "weight_vector": w[:len(diff_state)].tolist(),
        },
    }


def _canonicalize_pseudo_labels(
    value: Any,
    action_chunk: np.ndarray,
    *,
    state_t: np.ndarray | None = None,
    future_state: np.ndarray | None = None,
    label_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    use_v2 = state_t is not None and future_state is not None and label_params is not None
    if use_v2:
        params = label_params or {}
        computed = _compute_pseudo_labels_v2(
            action_chunk, state_t, future_state,
            tau_s=params.get("tau_s", DEFAULT_TAU_S),
            tau_fail=params.get("tau_fail", DEFAULT_TAU_FAIL),
            alpha_d=params.get("alpha_d", DEFAULT_ALPHA_D),
            beta_delta=params.get("beta_delta", DEFAULT_BETA_DELTA),
            gamma_region=params.get("gamma_region", DEFAULT_GAMMA_REGION),
        )
    else:
        computed = _compute_pseudo_labels(action_chunk)
    source = value if isinstance(value, dict) else {}
    pseudo = dict(source)
    for key, fallback in computed.items():
        if key not in pseudo:
            pseudo[key] = fallback
    return pseudo


def _build_region_target_15(pseudo_labels: dict[str, Any], target_len: int) -> tuple[list[float], str]:
    prior = _resample_vector(pseudo_labels.get("affected_region_prior"), target_len)
    mask = _resample_vector(pseudo_labels.get("correction_mask"), target_len)

    if prior is not None and mask is not None:
        merged = np.maximum(np.clip(prior, 0.0, 1.0), np.clip(mask, 0.0, 1.0)).astype(np.float32)
        source = "max(prior,mask)"
    elif prior is not None:
        merged = np.clip(prior, 0.0, 1.0).astype(np.float32)
        source = "affected_region_prior"
    elif mask is not None:
        merged = np.clip(mask, 0.0, 1.0).astype(np.float32)
        source = "correction_mask"
    else:
        merged = np.zeros((target_len,), dtype=np.float32)
        source = "zeros"

    merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
    return merged.tolist(), source


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _record_key(meta: dict[str, Any], sample_step: int | None = None) -> tuple[str, int, int] | None:
    dataset_name = _nonempty_string(meta.get("dataset_name"))
    trajectory_id = _safe_int(meta.get("trajectory_id"))
    step_value = sample_step if sample_step is not None else _safe_int(meta.get("sample_step"))
    if dataset_name is None or trajectory_id is None or step_value is None:
        return None
    return dataset_name, trajectory_id, int(step_value)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_future_state_index(path: Path) -> dict[tuple[str, int, int], np.ndarray]:
    index: dict[tuple[str, int, int], np.ndarray] = {}
    for record in _iter_jsonl(path):
        meta = record.get("meta")
        if not isinstance(meta, dict):
            continue
        key = _record_key(meta)
        if key is None:
            continue
        state = _flatten_vector(record.get("state_t", record.get("state")))
        if state is None:
            continue
        index[key] = state
    return index


def _compute_outcome_stats(emitted: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute AH-1~AH-3 statistics from emitted rows for stats output.

    Returns empty dict if no v2 outcome data is present (legacy mode).
    """
    if not emitted:
        return {}

    trigger_labels = []
    region_active_bins = []
    zero_delta_triggers = 0
    zero_delta_total = 0
    d_H_values = []
    has_v2 = False

    for row in emitted:
        pseudo = row.get("pseudo_labels", {})
        trigger = pseudo.get("trigger_label", 0)
        trigger_labels.append(trigger)

        region = row.get("region_target_15", [])
        if isinstance(region, list):
            active = sum(1 for v in region if isinstance(v, (int, float)) and v > 0.5)
            region_active_bins.append(active)

        delta_norms = pseudo.get("delta_action_norm", [])
        if isinstance(delta_norms, list):
            max_dn = max(delta_norms) if delta_norms else 0.0
        else:
            max_dn = float(delta_norms) if delta_norms else 0.0
        if max_dn < 1e-6:
            zero_delta_total += 1
            if trigger == 1:
                zero_delta_triggers += 1

        outcome_v2 = pseudo.get("_outcome_v2", {})
        if isinstance(outcome_v2, dict) and "d_H" in outcome_v2:
            has_v2 = True
            d_H_values.append(outcome_v2["d_H"])

    if not has_v2:
        return {}

    n = len(trigger_labels)
    trigger_positive_rate = float(sum(trigger_labels)) / n if n > 0 else 0.0
    mean_region_active = float(np.mean(region_active_bins)) if region_active_bins else 0.0
    zero_delta_false_trigger_rate = (
        float(zero_delta_triggers) / zero_delta_total if zero_delta_total > 0 else 0.0
    )

    stats: dict[str, Any] = {
        "outcome_label_version": "v0.9",
        "AH1_zero_delta_false_trigger_rate": zero_delta_false_trigger_rate,
        "AH1_zero_delta_samples": zero_delta_total,
        "AH2_trigger_positive_rate": trigger_positive_rate,
        "AH3_mean_region_active_bins": mean_region_active,
    }
    if d_H_values:
        d_arr = np.array(d_H_values)
        stats["d_H_P90"] = float(np.quantile(d_arr, 0.90))
        stats["d_H_mean"] = float(np.mean(d_arr))
        stats["d_H_max"] = float(np.max(d_arr))
    return stats


def _build_record(
    *,
    lang: str,
    meta: dict[str, Any],
    action_chunk: np.ndarray,
    state_t: np.ndarray,
    future_state: np.ndarray,
    future_state_step: int,
    horizon_steps: int,
    pseudo_labels: dict[str, Any],
    a_outputs: Any = None,
) -> dict[str, Any]:
    region_target_15, region_source = _build_region_target_15(pseudo_labels, CANONICAL_REGION_LEN)
    out = {
        "schema_version": PHASE0_SCHEMA_VERSION,
        "lang": lang,
        "meta": dict(meta),
        "horizon_steps": int(horizon_steps),
        "state_t": state_t.astype(np.float32).tolist(),
        "future_state": future_state.astype(np.float32).tolist(),
        "future_state_step": int(future_state_step),
        "action_chunk": action_chunk.astype(np.float32).tolist(),
        "remaining_chunk": action_chunk[1:].astype(np.float32).tolist(),
        "pseudo_labels": pseudo_labels,
        "region_target_15": region_target_15,
        "region_target_15_source": region_source,
    }
    if isinstance(a_outputs, dict):
        out["a_outputs"] = a_outputs
    return out


def _build_from_correction_jsonl(
    *,
    input_jsonl: Path,
    future_source_jsonl: Path,
    horizon_steps: int,
    label_params: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_rows = _iter_jsonl(input_jsonl)
    future_index = _load_future_state_index(future_source_jsonl)

    # Pass 1: validate rows and collect raw deviations for τ_s calibration
    valid_items: list[tuple[int, dict, np.ndarray, np.ndarray, np.ndarray]] = []
    dropped_missing_future = 0
    dropped_invalid = 0
    raw_deviations: list[float] = []

    for idx, record in enumerate(input_rows):
        meta = record.get("meta")
        if not isinstance(meta, dict):
            dropped_invalid += 1
            continue
        key = _record_key(meta)
        if key is None:
            dropped_invalid += 1
            continue
        dataset_name, trajectory_id, sample_step = key
        future_key = (dataset_name, trajectory_id, sample_step + horizon_steps)
        future_state = future_index.get(future_key)
        if future_state is None:
            dropped_missing_future += 1
            continue

        action_chunk = _to_2d_float32(record.get("action_chunk"))
        state_t = _flatten_vector(record.get("state_t", record.get("state")))
        if action_chunk is None or state_t is None:
            dropped_invalid += 1
            continue

        valid_items.append((idx, record, action_chunk, state_t, future_state))
        if label_params is not None:
            raw_deviations.append(_raw_weighted_deviation(state_t, future_state))

    # τ_s calibration (only in v2 mode)
    calibration_info: dict[str, Any] | None = None
    effective_params = label_params
    if label_params is not None and raw_deviations:
        user_tau_s = label_params.get("tau_s", DEFAULT_TAU_S)
        calibrated_tau_s, calibration_info = _calibrate_tau_s(raw_deviations, user_tau_s)
        effective_params = dict(label_params)
        effective_params["tau_s"] = calibrated_tau_s

    # Pass 2: compute labels with calibrated τ_s
    emitted: list[dict[str, Any]] = []
    for idx, record, action_chunk, state_t, future_state in valid_items:
        meta = record.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        key = _record_key(meta)
        if key is None:
            continue
        dataset_name, trajectory_id, sample_step = key

        pseudo_labels = _canonicalize_pseudo_labels(
            record.get("pseudo_labels"), action_chunk,
            state_t=state_t, future_state=future_state,
            label_params=effective_params,
        )
        output_meta = {
            "record_index": int(idx),
            "source_schema_version": meta.get("source_schema_version", record.get("schema_version", "legacy")),
            "dataset_name": dataset_name,
            "trajectory_id": trajectory_id,
            "sample_step": sample_step,
            "action_chunk_len": int(action_chunk.shape[0]),
            "action_dim": int(action_chunk.shape[1]),
            "state_dim": int(state_t.shape[0]),
        }
        emitted.append(
            _build_record(
                lang=str(record.get("lang", "")),
                meta=output_meta,
                action_chunk=action_chunk,
                state_t=state_t,
                future_state=future_state,
                future_state_step=sample_step + horizon_steps,
                horizon_steps=horizon_steps,
                pseudo_labels=pseudo_labels,
                a_outputs=record.get("a_outputs"),
            )
        )

    stats: dict[str, Any] = {
        "source_mode": "correction_jsonl",
        "input_jsonl": str(input_jsonl),
        "future_source_jsonl": str(future_source_jsonl),
        "rows_input": len(input_rows),
        "rows_emitted": len(emitted),
        "rows_dropped_missing_future": dropped_missing_future,
        "rows_dropped_invalid": dropped_invalid,
        "horizon_steps": int(horizon_steps),
    }
    stats.update(_compute_outcome_stats(emitted))
    if calibration_info is not None:
        stats["tau_s_calibration"] = calibration_info
    return emitted, stats


def _load_task_map(dataset_root: Path) -> dict[int, str]:
    task_path = dataset_root / "meta" / "tasks.jsonl"
    if not task_path.exists():
        return {}
    task_map: dict[int, str] = {}
    with task_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            task_index = _safe_int(obj.get("task_index"))
            task = _nonempty_string(obj.get("task"))
            if task_index is not None and task is not None:
                task_map[task_index] = task
    return task_map


def _build_from_dataset_root(
    *,
    dataset_root: Path,
    dataset_name: str,
    horizon_steps: int,
    action_chunk_len: int,
    sample_stride: int,
    action_key: str,
    state_key: str,
    max_samples: int | None,
    label_params: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "dataset-root mode requires pandas with a parquet engine (pyarrow or fastparquet)."
        ) from exc

    task_map = _load_task_map(dataset_root)
    parquet_paths = sorted((dataset_root / "data").rglob("episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet episodes found under: {dataset_root / 'data'}")

    # Pass 1: extract samples and collect raw deviations for τ_s calibration
    SampleTuple = tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int | None, str]
    valid_samples: list[SampleTuple] = []
    episodes_processed = 0
    skipped_short_episodes = 0
    raw_deviations: list[float] = []

    required_columns = [action_key, state_key, "episode_index", "frame_index", "task_index"]
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path, columns=required_columns)
        if len(df) == 0:
            continue
        episodes_processed += 1
        max_start = min(len(df) - action_chunk_len, len(df) - horizon_steps - 1)
        if max_start < 0:
            skipped_short_episodes += 1
            continue

        for start in range(0, max_start + 1, sample_stride):
            action_rows = [
                np.asarray(df.iloc[start + offset][action_key], dtype=np.float32).reshape(-1)
                for offset in range(action_chunk_len)
            ]
            action_chunk = np.stack(action_rows, axis=0).astype(np.float32)
            state_t = np.asarray(df.iloc[start][state_key], dtype=np.float32).reshape(-1)
            future_state = np.asarray(df.iloc[start + horizon_steps][state_key], dtype=np.float32).reshape(-1)
            episode_index = int(df.iloc[start]["episode_index"])
            frame_index = int(df.iloc[start]["frame_index"])
            task_index = _safe_int(df.iloc[start]["task_index"])
            lang = task_map.get(task_index if task_index is not None else -1, f"task_index={task_index}")

            valid_samples.append((action_chunk, state_t, future_state, episode_index, frame_index, task_index, lang))
            if label_params is not None:
                raw_deviations.append(_raw_weighted_deviation(state_t, future_state))

            if max_samples is not None and len(valid_samples) >= max_samples:
                break
        if max_samples is not None and len(valid_samples) >= max_samples:
            break

    # τ_s calibration
    calibration_info: dict[str, Any] | None = None
    effective_params = label_params
    if label_params is not None and raw_deviations:
        user_tau_s = label_params.get("tau_s", DEFAULT_TAU_S)
        calibrated_tau_s, calibration_info = _calibrate_tau_s(raw_deviations, user_tau_s)
        effective_params = dict(label_params)
        effective_params["tau_s"] = calibrated_tau_s

    # Pass 2: compute labels with calibrated τ_s
    emitted: list[dict[str, Any]] = []
    for action_chunk, state_t, future_state, episode_index, frame_index, task_index, lang in valid_samples:
        pseudo_labels = _canonicalize_pseudo_labels(
            None, action_chunk,
            state_t=state_t, future_state=future_state,
            label_params=effective_params,
        )
        meta = {
            "record_index": len(emitted),
            "source_schema_version": "lerobot_episode_v1",
            "dataset_name": dataset_name,
            "trajectory_id": episode_index,
            "sample_step": frame_index,
            "action_chunk_len": int(action_chunk.shape[0]),
            "action_dim": int(action_chunk.shape[1]),
            "state_dim": int(state_t.shape[0]),
        }
        emitted.append(
            _build_record(
                lang=lang,
                meta=meta,
                action_chunk=action_chunk,
                state_t=state_t,
                future_state=future_state,
                future_state_step=frame_index + horizon_steps,
                horizon_steps=horizon_steps,
                pseudo_labels=pseudo_labels,
            )
        )

    stats_final: dict[str, Any] = {
        "source_mode": "lerobot_dataset_root",
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_name,
        "rows_emitted": len(emitted),
        "episodes_processed": episodes_processed,
        "skipped_short_episodes": skipped_short_episodes,
        "horizon_steps": int(horizon_steps),
        "action_chunk_len": int(action_chunk_len),
        "sample_stride": int(sample_stride),
        "action_key": action_key,
        "state_key": state_key,
        "max_samples": None if max_samples is None else int(max_samples),
    }
    stats_final.update(_compute_outcome_stats(emitted))
    if calibration_info is not None:
        stats_final["tau_s_calibration"] = calibration_info
    return emitted, stats_final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase0 FASA dataset with future-state backfill.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=str, help="Existing correction JSONL with meta/state/action fields.")
    source.add_argument("--dataset-root", type=str, help="LeRobot-style dataset root with meta/ + data/*.parquet.")
    parser.add_argument("--future-source-jsonl", type=str, default=None, help="Optional JSONL keyed by meta for t+H state lookup.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Where to write the Phase0 JSONL.")
    parser.add_argument("--stats-json", type=str, default=None, help="Optional JSON stats output path.")
    parser.add_argument("--horizon-steps", type=int, default=4, help="Future-state horizon H.")
    parser.add_argument("--action-chunk-len", type=int, default=8, help="Chunk length when building from dataset root.")
    parser.add_argument("--sample-stride", type=int, default=4, help="Stride when scanning dataset root episodes.")
    parser.add_argument("--action-key", type=str, default="action.delta_joints", help="Action column for dataset-root mode.")
    parser.add_argument("--state-key", type=str, default="state.joints", help="State column for dataset-root mode.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional dataset_name override for dataset-root mode.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on emitted samples.")
    parser.add_argument("--tau-s", type=float, default=DEFAULT_TAU_S, help="State deviation normalization threshold (§3.2).")
    parser.add_argument("--tau-fail", type=float, default=DEFAULT_TAU_FAIL, help="Failure detection threshold on d_k (§3.3).")
    parser.add_argument("--alpha-d", type=float, default=DEFAULT_ALPHA_D, help="Trigger threshold on d_H (§3.4).")
    parser.add_argument("--beta-delta", type=float, default=DEFAULT_BETA_DELTA, help="Delta-action outcome mixing coefficient (§3.4).")
    parser.add_argument("--gamma-region", type=float, default=DEFAULT_GAMMA_REGION, help="Region prior outcome mixing coefficient (§3.4).")
    parser.add_argument("--legacy-labels", action="store_true", help="Use legacy delta-only pseudo labels instead of outcome v0.9.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon_steps <= 0:
        raise ValueError("--horizon-steps must be > 0 for future-state backfill.")
    if args.action_chunk_len <= 1:
        raise ValueError("--action-chunk-len must be > 1.")
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be > 0.")

    label_params: dict[str, float] | None = None
    if not args.legacy_labels:
        label_params = {
            "tau_s": args.tau_s,
            "tau_fail": args.tau_fail,
            "alpha_d": args.alpha_d,
            "beta_delta": args.beta_delta,
            "gamma_region": args.gamma_region,
        }

    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.input_jsonl:
        input_jsonl = Path(args.input_jsonl).expanduser().resolve()
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
        future_source_jsonl = Path(args.future_source_jsonl).expanduser().resolve() if args.future_source_jsonl else input_jsonl
        if not future_source_jsonl.exists():
            raise FileNotFoundError(f"Future-source JSONL not found: {future_source_jsonl}")
        rows, stats = _build_from_correction_jsonl(
            input_jsonl=input_jsonl,
            future_source_jsonl=future_source_jsonl,
            horizon_steps=args.horizon_steps,
            label_params=label_params,
        )
    else:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        rows, stats = _build_from_dataset_root(
            dataset_root=dataset_root,
            dataset_name=args.dataset_name or dataset_root.name,
            horizon_steps=args.horizon_steps,
            action_chunk_len=args.action_chunk_len,
            sample_stride=args.sample_stride,
            action_key=args.action_key,
            state_key=args.state_key,
            max_samples=args.max_samples,
            label_params=label_params,
        )

    if not rows:
        raise RuntimeError(
            "No Phase0 rows were emitted. Provide a continuous future-state source "
            "or lower horizon/sample-stride constraints."
        )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = dict(stats)
    stats["output_jsonl"] = str(output_jsonl)
    if args.stats_json:
        stats_path = Path(args.stats_json).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
