from __future__ import annotations

from typing import Any

import numpy as np
import torch


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


def build_optional_hook_targets(
    examples: list[dict],
    *,
    chunk_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Build tensor targets/masks from sample-level pseudo labels."""
    batch_size = len(examples)
    targets = {
        "risk_score": torch.zeros((batch_size,), device=device, dtype=dtype),
        "risk_mask": torch.zeros((batch_size,), device=device, dtype=torch.bool),
        "trigger_label": torch.zeros((batch_size,), device=device, dtype=dtype),
        "trigger_mask": torch.zeros((batch_size,), device=device, dtype=torch.bool),
        "delta_action_norm": torch.zeros((batch_size,), device=device, dtype=dtype),
        "delta_mask": torch.zeros((batch_size,), device=device, dtype=torch.bool),
        "correction_mask": torch.zeros((batch_size, chunk_len), device=device, dtype=dtype),
        "correction_mask_mask": torch.zeros((batch_size,), device=device, dtype=torch.bool),
        "region_prior": torch.zeros((batch_size, chunk_len), device=device, dtype=dtype),
        "region_prior_mask": torch.zeros((batch_size,), device=device, dtype=torch.bool),
    }

    for i, example in enumerate(examples):
        pseudo = example.get("pseudo_labels")
        if not isinstance(pseudo, dict):
            continue

        risk = _safe_float(pseudo.get("risk_score"))
        if risk is not None:
            targets["risk_score"][i] = risk
            targets["risk_mask"][i] = True

        trigger = _safe_float(pseudo.get("trigger_label"))
        if trigger is not None:
            targets["trigger_label"][i] = trigger
            targets["trigger_mask"][i] = True

        delta_norm = _safe_float(pseudo.get("delta_action_norm"))
        if delta_norm is not None:
            targets["delta_action_norm"][i] = delta_norm
            targets["delta_mask"][i] = True

        correction_mask = _coerce_vector(pseudo.get("correction_mask"), target_len=chunk_len)
        if correction_mask is not None:
            targets["correction_mask"][i] = torch.as_tensor(
                correction_mask,
                device=device,
                dtype=dtype,
            )
            targets["correction_mask_mask"][i] = True

        region_prior = _coerce_vector(pseudo.get("affected_region_prior"), target_len=chunk_len)
        if region_prior is not None:
            targets["region_prior"][i] = torch.as_tensor(
                region_prior,
                device=device,
                dtype=dtype,
            )
            targets["region_prior_mask"][i] = True

    return targets
