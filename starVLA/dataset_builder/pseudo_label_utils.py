"""Pseudo-label helpers for the correction dataset builder."""

from __future__ import annotations

import numpy as np


def compute_correction_pseudo_labels(action_chunk: np.ndarray) -> dict:
    """
    Build simple, deterministic pseudo labels from action chunk.

    action_chunk: [T, D]
    outputs align to remaining chunk length H_rem = T - 1.
    """
    if action_chunk.ndim != 2 or action_chunk.shape[0] <= 0 or action_chunk.shape[1] <= 0:
        raise ValueError(f"action_chunk must be rank-2 with positive dims, got {action_chunk.shape}")

    if action_chunk.shape[0] == 1:
        delta_norm = np.zeros((0,), dtype=np.float32)
    else:
        delta_norm = np.linalg.norm(np.diff(action_chunk, axis=0), axis=1).astype(np.float32)

    if delta_norm.size == 0:
        threshold = 0.0
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

