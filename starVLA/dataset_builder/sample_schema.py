"""Schema definitions and validators for correction pseudo-label dataset entries."""

from __future__ import annotations

from typing import Any


CORRECTION_SCHEMA_VERSION = "p1_correction_dataset_v1"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_2d_list(name: str, value: Any, errors: list[str]) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) == 0:
        errors.append(f"`{name}` must be a non-empty 2D list.")
        return None
    if not all(isinstance(row, list) for row in value):
        errors.append(f"`{name}` must be a 2D list.")
        return None
    row_lens = {len(row) for row in value}
    if len(row_lens) != 1:
        errors.append(f"`{name}` rows must have equal length.")
        return None
    cols = next(iter(row_lens))
    if cols <= 0:
        errors.append(f"`{name}` second dimension must be > 0.")
        return None
    for row in value:
        if not all(_is_number(v) for v in row):
            errors.append(f"`{name}` contains non-numeric values.")
            return None
    return (len(value), cols)


def validate_correction_entry(entry: dict[str, Any]) -> list[str]:
    """Return validation errors for one correction-dataset entry."""
    errors: list[str] = []
    if not isinstance(entry, dict):
        return ["entry must be dict"]

    for key in ["schema_version", "lang", "action_chunk", "remaining_chunk", "pseudo_labels", "meta"]:
        if key not in entry:
            errors.append(f"missing key `{key}`")

    if errors:
        return errors

    if entry.get("schema_version") != CORRECTION_SCHEMA_VERSION:
        errors.append(
            f"`schema_version` mismatch: got {entry.get('schema_version')!r}, "
            f"expected {CORRECTION_SCHEMA_VERSION!r}"
        )

    if not isinstance(entry.get("lang"), str):
        errors.append("`lang` must be string.")

    action_shape = _validate_2d_list("action_chunk", entry.get("action_chunk"), errors)
    remaining_shape = _validate_2d_list("remaining_chunk", entry.get("remaining_chunk"), errors)

    labels = entry.get("pseudo_labels")
    if not isinstance(labels, dict):
        errors.append("`pseudo_labels` must be dict.")
    else:
        for key in ["trigger_label", "risk_score", "affected_region_prior", "correction_mask"]:
            if key not in labels:
                errors.append(f"`pseudo_labels.{key}` is required.")
        trigger_label = labels.get("trigger_label")
        if trigger_label not in (0, 1):
            errors.append("`pseudo_labels.trigger_label` must be 0 or 1.")
        risk_score = labels.get("risk_score")
        if not _is_number(risk_score):
            errors.append("`pseudo_labels.risk_score` must be numeric.")
        prior = labels.get("affected_region_prior")
        if not isinstance(prior, list):
            errors.append("`pseudo_labels.affected_region_prior` must be list.")
        elif not all(_is_number(v) for v in prior):
            errors.append("`pseudo_labels.affected_region_prior` must be numeric list.")
        mask = labels.get("correction_mask")
        if not isinstance(mask, list):
            errors.append("`pseudo_labels.correction_mask` must be list.")
        elif not all(v in (0, 1) for v in mask):
            errors.append("`pseudo_labels.correction_mask` must contain only 0/1.")

        if remaining_shape is not None and isinstance(prior, list) and len(prior) != remaining_shape[0]:
            errors.append(
                f"`pseudo_labels.affected_region_prior` length mismatch: got {len(prior)}, "
                f"expected {remaining_shape[0]}"
            )
        if remaining_shape is not None and isinstance(mask, list) and len(mask) != remaining_shape[0]:
            errors.append(
                f"`pseudo_labels.correction_mask` length mismatch: got {len(mask)}, expected {remaining_shape[0]}"
            )

    meta = entry.get("meta")
    if not isinstance(meta, dict):
        errors.append("`meta` must be dict.")
    else:
        for key in ["action_chunk_len", "action_dim", "source_schema_version"]:
            if key not in meta:
                errors.append(f"`meta.{key}` is required.")
        if action_shape is not None:
            if meta.get("action_chunk_len") != action_shape[0]:
                errors.append(
                    f"`meta.action_chunk_len` mismatch: got {meta.get('action_chunk_len')}, expected {action_shape[0]}"
                )
            if meta.get("action_dim") != action_shape[1]:
                errors.append(
                    f"`meta.action_dim` mismatch: got {meta.get('action_dim')}, expected {action_shape[1]}"
                )

    return errors

