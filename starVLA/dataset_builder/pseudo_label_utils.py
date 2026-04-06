"""Pseudo-label helpers for the correction dataset builder."""

from __future__ import annotations

import numpy as np

# Zero-like risk floor used for deterministic consistency checks.
DEFAULT_ZERO_EPS = 1e-8
# Trigger threshold for correction branch activation.
DEFAULT_TRIGGER_THRESHOLD = 0.1
# Keep only higher-risk region as correction mask.
DEFAULT_MASK_QUANTILE = 0.75
# Avoid very small mask threshold when quantile is too low.
DEFAULT_RELATIVE_MASK_FLOOR = 0.4


def _compute_delta_norm(action_chunk: np.ndarray) -> np.ndarray:
    """Compute per-step action delta norm for pseudo-labeling.

    For common 7D action layout (xyz, rpy, gripper), use motion channels
    (`[:-1]`) to avoid binary gripper toggles dominating risk score.
    """
    if action_chunk.shape[0] <= 1:
        return np.zeros((0,), dtype=np.float32)

    delta = np.diff(action_chunk, axis=0).astype(np.float32)
    if delta.shape[1] >= 2:
        delta_for_risk = delta[:, :-1]
    else:
        delta_for_risk = delta
    return np.linalg.norm(delta_for_risk, axis=1).astype(np.float32)


def compute_correction_pseudo_labels(
    action_chunk: np.ndarray,
    *,
    zero_eps: float = DEFAULT_ZERO_EPS,
    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    mask_quantile: float = DEFAULT_MASK_QUANTILE,
    relative_mask_floor: float = DEFAULT_RELATIVE_MASK_FLOOR,
) -> dict:
    """
    Build deterministic pseudo labels from action chunk.

    action_chunk: [T, D]
    outputs align to remaining chunk length H_rem = T - 1.
    """
    if action_chunk.ndim != 2 or action_chunk.shape[0] <= 0 or action_chunk.shape[1] <= 0:
        raise ValueError(f"action_chunk must be rank-2 with positive dims, got {action_chunk.shape}")

    delta_norm = _compute_delta_norm(action_chunk)
    rem_len = int(delta_norm.shape[0])

    if rem_len == 0:
        return {
            "trigger_label": 0,
            "risk_score": 0.0,
            "affected_region_prior": [],
            "correction_mask": [],
            "delta_action_norm": [],
        }

    risk_score = float(np.max(delta_norm))
    if risk_score <= float(zero_eps):
        return {
            "trigger_label": 0,
            "risk_score": 0.0,
            "affected_region_prior": np.zeros((rem_len,), dtype=np.float32).tolist(),
            "correction_mask": np.zeros((rem_len,), dtype=np.int32).tolist(),
            "delta_action_norm": delta_norm.tolist(),
        }

    denom = float(np.sum(delta_norm))
    if denom > float(zero_eps):
        affected_region_prior = (delta_norm / denom).astype(np.float32)
    else:
        affected_region_prior = np.zeros((rem_len,), dtype=np.float32)

    quantile_threshold = float(np.quantile(delta_norm, float(mask_quantile)))
    effective_threshold = max(
        quantile_threshold,
        float(zero_eps),
        risk_score * float(relative_mask_floor),
    )
    correction_mask = (delta_norm >= effective_threshold).astype(np.int32)
    if correction_mask.max(initial=0) == 0:
        correction_mask[int(np.argmax(delta_norm))] = 1

    trigger_label = int(risk_score >= float(trigger_threshold))
    if trigger_label == 0:
        correction_mask = np.zeros_like(correction_mask, dtype=np.int32)
        affected_region_prior = np.zeros_like(affected_region_prior, dtype=np.float32)

    return {
        "trigger_label": trigger_label,
        "risk_score": risk_score,
        "affected_region_prior": affected_region_prior.tolist(),
        "correction_mask": correction_mask.tolist(),
        "delta_action_norm": delta_norm.tolist(),
    }
