#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_REGION_LEN = 15
REQUIRED_EMBED_LEN = 16
DEFAULT_VERSION = "fasa_v1"
DEFAULT_SOURCE = "fasa/main"
SOURCE_FALLBACK = "fasa/fallback"
SOURCE_INTERP15 = "fasa/interp15"
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


def _prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = max(eps, min(1.0 - eps, prob))
    return float(math.log(clipped / (1.0 - clipped)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return float(1.0 / (1.0 + z))
    z = math.exp(value)
    return float(z / (1.0 + z))


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


def _coerce_vector(value: Any, target_len: int) -> np.ndarray | None:
    if target_len <= 0:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size >= target_len:
        return arr[:target_len]
    padded = np.zeros((target_len,), dtype=np.float32)
    padded[: arr.size] = arr
    return padded


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


def _record_key(record: dict[str, Any]) -> tuple[str, int, int] | None:
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return None
    dataset_name = str(meta.get("dataset_name", "")).strip()
    trajectory_id = _safe_int(meta.get("trajectory_id"))
    sample_step = _safe_int(meta.get("sample_step"))
    if not dataset_name or trajectory_id is None or sample_step is None:
        return None
    return dataset_name, trajectory_id, sample_step


def _region_probs(record: dict[str, Any], pseudo: dict[str, Any], region_len: int) -> tuple[list[float], str, bool, bool]:
    region = record.get("region_target_15")
    if isinstance(region, list) and len(region) == region_len:
        vals = [_safe_float(item) for item in region]
        if all(item is not None for item in vals):
            return [_clip01(float(item)) for item in vals if item is not None], "region_target_15", False, True

    prior = _finite_vector(pseudo.get("affected_region_prior"))
    mask = _finite_vector(pseudo.get("correction_mask"))

    if prior and mask:
        rp = _resample(prior, region_len)
        rm = _resample(mask, region_len)
        return [_clip01(max(a, b)) for a, b in zip(rp, rm)], "max(prior,mask)", True, True
    if prior:
        return [_clip01(x) for x in _resample(prior, region_len)], "affected_region_prior", True, True
    if mask:
        return [_clip01(x) for x in _resample(mask, region_len)], "correction_mask", True, True
    return [0.0] * region_len, "zeros", True, False


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(max(0.0, var))
    return mean, std, min(values), max(values)


def _dynamic_embedding(
    record: dict[str, Any], risk_pred: float, trigger_logit: float, delta_pred: float, embed_len: int
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
    if len(out) < embed_len:
        out.extend([0.0] * (embed_len - len(out)))
    return out[:embed_len], not has_signal, has_signal


def _get_action_window(record: dict[str, Any]) -> np.ndarray:
    for key in ("remaining_chunk", "action_chunk"):
        value = record.get(key)
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            continue
        if arr.ndim == 2 and arr.size > 0:
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.zeros((1, 1), dtype=np.float32)


def _get_state_vector(record: dict[str, Any]) -> np.ndarray:
    for key in ("state", "state_t"):
        try:
            arr = np.asarray(record.get(key), dtype=np.float32)
        except Exception:
            arr = None
        if arr is not None and arr.size > 0:
            return np.nan_to_num(arr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.zeros((1,), dtype=np.float32)


def _derive_surrogate_outputs(
    record: dict[str, Any],
    *,
    region_len: int,
    embedding_dim: int,
    version: str,
    source: str,
) -> dict[str, Any]:
    action_window = _get_action_window(record)
    state_vec = _get_state_vector(record)

    action_body = action_window[:, :-1] if action_window.shape[1] > 1 else action_window
    motion = np.linalg.norm(action_body, axis=1).astype(np.float32)
    velocity = (
        np.linalg.norm(np.diff(action_body, axis=0), axis=1).astype(np.float32)
        if action_body.shape[0] > 1
        else np.zeros((1,), dtype=np.float32)
    )
    gripper = action_window[:, -1].astype(np.float32) if action_window.shape[1] > 0 else np.zeros((1,), dtype=np.float32)

    motion_mean = float(motion.mean()) if motion.size else 0.0
    motion_std = float(motion.std()) if motion.size else 0.0
    motion_max = float(motion.max()) if motion.size else 0.0
    motion_tail = float(motion[-1]) if motion.size else 0.0
    velocity_mean = float(velocity.mean()) if velocity.size else 0.0
    velocity_std = float(velocity.std()) if velocity.size else 0.0
    velocity_max = float(velocity.max()) if velocity.size else 0.0
    tail_minus_head = float(motion[-1] - motion[0]) if motion.size > 1 else 0.0
    gripper_mean = float(gripper.mean()) if gripper.size else 0.0
    gripper_last = float(gripper[-1]) if gripper.size else 0.0
    xyz_mean_abs = float(np.abs(action_window[:, :3]).mean()) if action_window.shape[1] >= 3 else motion_mean
    rot_mean_abs = float(np.abs(action_window[:, 3:6]).mean()) if action_window.shape[1] >= 6 else motion_std
    state_mean_abs = float(np.abs(state_vec).mean()) if state_vec.size else 0.0
    state_std = float(state_vec.std()) if state_vec.size else 0.0
    chunk_len_norm = float(action_window.shape[0] / max(1, region_len))
    action_dim_norm = float(action_window.shape[1] / 7.0) if action_window.shape[1] > 0 else 0.0

    embedding_features = np.asarray(
        [
            motion_mean,
            motion_std,
            motion_max,
            motion_tail,
            velocity_mean,
            velocity_std,
            velocity_max,
            tail_minus_head,
            gripper_mean,
            gripper_last,
            xyz_mean_abs,
            rot_mean_abs,
            state_mean_abs,
            state_std,
            chunk_len_norm,
            action_dim_norm,
        ],
        dtype=np.float32,
    )
    if embedding_features.size < embedding_dim:
        embedding_features = np.pad(embedding_features, (0, embedding_dim - embedding_features.size), mode="constant")
    else:
        embedding_features = embedding_features[:embedding_dim]

    risk_raw = (
        3.5 * motion_max
        + 2.0 * velocity_mean
        + 1.5 * abs(tail_minus_head)
        + 0.25 * state_mean_abs
        - 0.45
    )
    risk_pred = _sigmoid(float(risk_raw))
    trigger_logit = float(2.75 * (risk_pred - 0.5) + 1.5 * tail_minus_head + 0.25 * gripper_last)
    delta_pred = float(max(0.0, 0.7 * motion_max + 0.3 * velocity_max))

    region_base = motion
    if velocity.size:
        aligned_velocity = np.pad(velocity, (1, 0), mode="edge")[: motion.size]
        region_base = 0.7 * motion + 0.3 * aligned_velocity
    region_base = np.nan_to_num(region_base, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if region_base.size == 0:
        region_base = np.zeros((1,), dtype=np.float32)
    region_min = float(region_base.min())
    region_max = float(region_base.max())
    if region_max - region_min < 1e-6:
        region_prob = np.full((region_len,), min(0.95, max(0.05, 0.2 + 0.6 * risk_pred)), dtype=np.float32)
    else:
        scaled = (region_base - region_min) / max(region_max - region_min, 1e-6)
        resampled = _resample_vector(scaled, region_len)
        assert resampled is not None
        region_prob = np.clip(0.05 + 0.9 * resampled, 1e-4, 1.0 - 1e-4).astype(np.float32)

    return {
        "risk_pred": float(risk_pred),
        "trigger_logit": float(trigger_logit),
        "delta_pred": float(delta_pred),
        "region_logits": [_prob_to_logit(float(v)) for v in region_prob],
        "dynamic_embedding": embedding_features.astype(np.float32).tolist(),
        "version": version,
        "source": source,
    }


def _build_default_payload(
    record: dict[str, Any], *, version: str, source: str, region_len: int, embed_len: int
) -> tuple[dict[str, Any], dict[str, Any]]:
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

    region_probs, region_source, region_fallback, has_region_signal = _region_probs(record, pseudo, region_len)
    if region_fallback:
        fallback_types.add("region")
    if not has_region_signal:
        missing_fields.append("region supervision")

    region_logits = [_logit(_clip01(p)) for p in region_probs]
    dynamic_embedding, embed_fallback, has_embed_signal = _dynamic_embedding(
        record, risk_pred, trigger_logit, delta_pred, embed_len
    )
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
        "used_external": False,
        "used_external_fallback": False,
        "used_interp15": False,
        "external_source": "",
    }
    return payload, row_meta


def _pick_payload(record: dict[str, Any]) -> dict[str, Any]:
    nested = record.get("a_outputs")
    if isinstance(nested, dict):
        return nested
    return record


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no}: expected dict JSON object")
            rows.append(record)
    return rows


def _load_external_predictions(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in _load_jsonl(path):
        key = _record_key(record)
        if key is not None:
            index[key] = record
    return index


def _merge_external_payload(
    record: dict[str, Any],
    *,
    default_payload: dict[str, Any],
    row_meta: dict[str, Any],
    external_record: dict[str, Any] | None,
    region_len: int,
    embed_len: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(external_record, dict):
        return default_payload, row_meta

    external_payload = _pick_payload(external_record)
    if not isinstance(external_payload, dict):
        return default_payload, row_meta

    merged = dict(default_payload)
    row_meta = dict(row_meta)
    row_meta["used_external"] = True
    row_meta["external_source"] = str(external_payload.get("source") or "")

    external_fallback = False
    used_interp15 = False

    risk_pred = _safe_float(external_payload.get("risk_pred"))
    if risk_pred is not None:
        merged["risk_pred"] = float(risk_pred)
    else:
        external_fallback = True

    trigger_logit = _safe_float(external_payload.get("trigger_logit"))
    if trigger_logit is None:
        trigger_prob = _safe_float(external_payload.get("trigger_prob"))
        if trigger_prob is not None:
            trigger_logit = _prob_to_logit(trigger_prob)
    if trigger_logit is not None:
        merged["trigger_logit"] = float(trigger_logit)
    else:
        external_fallback = True

    delta_pred = _safe_float(external_payload.get("delta_pred"))
    if delta_pred is not None:
        merged["delta_pred"] = float(max(0.0, delta_pred))
    else:
        external_fallback = True

    raw_region_logits = external_payload.get("region_logits")
    region_logits = _resample_vector(raw_region_logits, region_len)
    if region_logits is not None:
        merged["region_logits"] = region_logits.astype(np.float32).tolist()
        if isinstance(raw_region_logits, (list, tuple)) and len(raw_region_logits) != region_len:
            used_interp15 = True
    else:
        external_fallback = True

    dynamic_embedding = _coerce_vector(external_payload.get("dynamic_embedding"), embed_len)
    if dynamic_embedding is not None:
        merged["dynamic_embedding"] = dynamic_embedding.astype(np.float32).tolist()
    else:
        external_fallback = True

    version = str(external_payload.get("version") or merged.get("version") or DEFAULT_VERSION).strip()
    source = str(external_payload.get("source") or merged.get("source") or DEFAULT_SOURCE).strip()
    if not version:
        version = DEFAULT_VERSION
    if not source:
        source = DEFAULT_SOURCE

    if external_fallback:
        source = SOURCE_FALLBACK
    elif used_interp15:
        source = SOURCE_INTERP15

    merged["version"] = version
    merged["source"] = source
    row_meta["used_external_fallback"] = external_fallback
    row_meta["used_interp15"] = used_interp15
    row_meta["row_with_fallback_any"] = row_meta["row_with_fallback_any"] or external_fallback
    return merged, row_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline FASA a_outputs JSONL from Phase0 JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Input Phase0 JSONL path.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path with sidecar a_outputs payload.")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Payload contract version.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Payload source tag.")
    parser.add_argument("--sample-check-json", default=None, help="Optional sample check output JSON.")
    parser.add_argument("--summary-json", default=None, help="Optional path to write producer summary JSON.")
    parser.add_argument("--stats-json", default=None, help="Alias of --summary-json for compatibility.")
    parser.add_argument("--max-samples", type=int, default=5, help="Max sample rows written into sample check.")
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail with non-zero exit when key supervision fields are missing.",
    )
    parser.add_argument("--max-print-missing", type=int, default=5, help="Max missing examples stored in summary.")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=REQUIRED_EMBED_LEN,
        help="Dynamic embedding dimension. Default preserves current contract.",
    )
    parser.add_argument(
        "--region-len",
        type=int,
        default=REQUIRED_REGION_LEN,
        help="Region logits length. Default preserves current contract.",
    )
    parser.add_argument(
        "--vdpm-jsonl",
        default="",
        help="Optional JSONL of precomputed sidecar/VDPM outputs keyed by meta.{dataset_name,trajectory_id,sample_step}.",
    )
    parser.add_argument(
        "--overwrite-existing-a-outputs",
        action="store_true",
        help="Overwrite record['a_outputs'] if already present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    rows = _load_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    external_index: dict[tuple[str, int, int], dict[str, Any]] = {}
    external_path = ""
    if args.vdpm_jsonl:
        external_path = str(Path(args.vdpm_jsonl).expanduser().resolve())
        external_index = _load_external_predictions(Path(external_path))

    rows_total = 0
    rows_updated = 0
    rows_kept_existing = 0
    rows_missing_meta = 0
    rows_with_fallback_any = 0
    rows_with_missing_pseudo_labels = 0
    rows_from_external_vdpm = 0
    rows_with_external_fallback = 0
    rows_with_interp15 = 0
    fallback_counts = {key: 0 for key in FALLBACK_TYPES}
    strict_missing_failed_rows = 0
    strict_missing_examples: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as fout:
        for line_no, record in enumerate(rows, start=1):
            rows_total += 1
            if not isinstance(record.get("meta"), dict):
                rows_missing_meta += 1

            if isinstance(record.get("a_outputs"), dict) and not args.overwrite_existing_a_outputs:
                rows_kept_existing += 1
                out_row = dict(record)
                out_row.setdefault("line_no", line_no)
                fout.write(json.dumps(out_row, ensure_ascii=False, allow_nan=False) + "\n")
                continue

            default_payload, row_meta = _build_default_payload(
                record,
                version=args.version,
                source=args.source,
                region_len=max(1, int(args.region_len)),
                embed_len=max(1, int(args.embedding_dim)),
            )

            if row_meta["row_with_missing_pseudo_labels"]:
                rows_with_missing_pseudo_labels += 1

            record_key = _record_key(record)
            external_record = external_index.get(record_key) if record_key is not None else None
            payload, row_meta = _merge_external_payload(
                record,
                default_payload=default_payload,
                row_meta=row_meta,
                external_record=external_record,
                region_len=max(1, int(args.region_len)),
                embed_len=max(1, int(args.embedding_dim)),
            )

            if row_meta["used_external"]:
                rows_from_external_vdpm += 1
            if row_meta["used_external_fallback"]:
                rows_with_external_fallback += 1
            if row_meta["used_interp15"]:
                rows_with_interp15 += 1

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
                        "used_external": row_meta["used_external"],
                        "used_external_fallback": row_meta["used_external_fallback"],
                        "used_interp15": row_meta["used_interp15"],
                    }
                )

            out_row = dict(record)
            out_row["line_no"] = line_no
            out_row["a_outputs"] = payload
            fout.write(json.dumps(out_row, ensure_ascii=False, allow_nan=False) + "\n")
            rows_updated += 1

    rows_with_fallback_any_ratio = float(rows_with_fallback_any / max(1, rows_total))
    strict_missing_pass = strict_missing_failed_rows == 0

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "external_vdpm_path": external_path,
        "rows_total": rows_total,
        "rows_updated": rows_updated,
        "rows_kept_existing": rows_kept_existing,
        "rows_missing_meta": rows_missing_meta,
        "rows_with_fallback_any": rows_with_fallback_any,
        "rows_with_fallback_any_ratio": rows_with_fallback_any_ratio,
        "rows_with_missing_pseudo_labels": rows_with_missing_pseudo_labels,
        "rows_from_external_vdpm": rows_from_external_vdpm,
        "rows_with_external_fallback": rows_with_external_fallback,
        "rows_with_interp15": rows_with_interp15,
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
            "region_logits_len": max(1, int(args.region_len)),
            "dynamic_embedding_len": max(1, int(args.embedding_dim)),
            "version": args.version,
            "source": args.source,
        },
    }

    if args.sample_check_json:
        sample_path = Path(args.sample_check_json).expanduser().resolve()
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_json_path = args.summary_json or args.stats_json
    if summary_json_path:
        summary_path = Path(summary_json_path).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict_missing and not strict_missing_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
