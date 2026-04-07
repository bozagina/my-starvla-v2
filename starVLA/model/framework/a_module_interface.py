from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


def _cfg_get(cfg_obj, key: str, default=None):
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, "get"):
        try:
            return cfg_obj.get(key, default)
        except Exception:
            pass
    return getattr(cfg_obj, key, default)


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _coerce_vector(value: Any, target_len: int) -> torch.Tensor | None:
    if target_len <= 0:
        return None
    try:
        arr = torch.as_tensor(value, dtype=torch.float32)
    except Exception:
        return None
    if arr.numel() == 0:
        return None
    arr = arr.reshape(-1)
    arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.numel() >= target_len:
        return arr[:target_len]
    padded = torch.zeros((target_len,), dtype=torch.float32)
    padded[: arr.numel()] = arr
    return padded


@dataclass
class AModulePredictions:
    risk_pred: torch.Tensor
    trigger_logit: torch.Tensor
    delta_pred: torch.Tensor
    region_logits: torch.Tensor
    debug_metrics: dict[str, float]


class LiteAModuleInterface:
    """Use in-graph lightweight heads for A/corrective predictions."""

    mode = "lite"

    def __init__(
        self,
        *,
        a_risk_head,
        a_trigger_head,
        corrective_delta_head,
        corrective_region_head,
    ) -> None:
        self.a_risk_head = a_risk_head
        self.a_trigger_head = a_trigger_head
        self.corrective_delta_head = corrective_delta_head
        self.corrective_region_head = corrective_region_head

    def predict(
        self,
        *,
        pooled_hidden: torch.Tensor,
        examples: list[dict],
        action_loss: torch.Tensor,
        chunk_len: int,
    ) -> AModulePredictions:
        _ = examples, chunk_len
        pooled_hidden = pooled_hidden.to(dtype=action_loss.dtype)
        return AModulePredictions(
            risk_pred=self.a_risk_head(pooled_hidden).squeeze(-1),
            trigger_logit=self.a_trigger_head(pooled_hidden).squeeze(-1),
            delta_pred=self.corrective_delta_head(pooled_hidden).squeeze(-1),
            region_logits=self.corrective_region_head(pooled_hidden),
            debug_metrics={"debug/a_module_mode_lite": 1.0},
        )


class StandaloneAModuleInterface:
    """Read predictions emitted by an external A module from per-sample payload."""

    mode = "standalone"

    def __init__(
        self,
        *,
        chunk_len: int,
        input_key: str = "a_outputs",
        allow_pseudo_labels_fallback: bool = False,
        strict_missing: bool = False,
        fallback_to_lite=None,
        strict_output_contract: bool = False,
        expected_region_len: int | None = None,
        expected_embedding_dim: int | None = None,
    ) -> None:
        self.chunk_len = int(chunk_len)
        self.input_key = str(input_key)
        self.allow_pseudo_labels_fallback = bool(allow_pseudo_labels_fallback)
        self.strict_missing = bool(strict_missing)
        self.fallback_to_lite = fallback_to_lite
        self.strict_output_contract = bool(strict_output_contract)
        self.expected_region_len = int(expected_region_len) if expected_region_len else None
        self.expected_embedding_dim = int(expected_embedding_dim) if expected_embedding_dim else None

    def _extract_source(self, example: dict) -> dict:
        source = example.get(self.input_key)
        if isinstance(source, dict):
            return source
        # Optional fallback for migration. Keep disabled by default to avoid label leakage.
        pseudo = example.get("pseudo_labels")
        if self.allow_pseudo_labels_fallback and isinstance(pseudo, dict):
            return pseudo
        return {}

    def _prediction_from_examples(
        self,
        *,
        examples: list[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[AModulePredictions, bool]:
        batch_size = len(examples)
        risk_pred = torch.zeros((batch_size,), device=device, dtype=dtype)
        trigger_logit = torch.zeros((batch_size,), device=device, dtype=dtype)
        delta_pred = torch.zeros((batch_size,), device=device, dtype=dtype)
        region_logits = torch.zeros((batch_size, self.chunk_len), device=device, dtype=dtype)

        any_signal = False
        fallback_used = 0
        rows_with_payload = 0
        rows_with_all_fields = 0
        risk_found = 0
        trigger_found = 0
        delta_found = 0
        region_found = 0
        region_input_present = 0
        region_len_mismatch = 0
        embedding_present = 0
        embedding_len_mismatch = 0
        risk_sum = 0.0
        delta_sum = 0.0
        trigger_pos = 0

        for i, example in enumerate(examples):
            source = self._extract_source(example)
            has_payload = bool(isinstance(source, dict) and len(source) > 0)
            if has_payload:
                rows_with_payload += 1
            row_has_all_fields = True

            if has_payload:
                embedding_raw = source.get("dynamic_embedding")
                if isinstance(embedding_raw, (list, tuple)):
                    embedding_present += 1
                    if self.expected_embedding_dim is not None and len(embedding_raw) != self.expected_embedding_dim:
                        embedding_len_mismatch += 1
                        if self.strict_output_contract:
                            raise ValueError(
                                "Standalone A output contract violation: "
                                f"dynamic_embedding len={len(embedding_raw)} "
                                f"!= expected_embedding_dim={self.expected_embedding_dim}"
                            )

            risk_value = _safe_float(source.get("risk_pred", source.get("risk_score")))
            if risk_value is not None:
                risk_pred[i] = risk_value
                any_signal = True
                risk_found += 1
                risk_sum += float(risk_value)
            else:
                fallback_used += 1
                row_has_all_fields = False

            trigger_logit_value = _safe_float(source.get("trigger_logit"))
            if trigger_logit_value is None:
                trigger_prob = _safe_float(source.get("trigger_prob", source.get("trigger_label")))
                if trigger_prob is not None:
                    prob = max(1e-6, min(1.0 - 1e-6, trigger_prob))
                    trigger_logit_value = float(torch.logit(torch.tensor(prob)).item())
            if trigger_logit_value is not None:
                trigger_logit[i] = trigger_logit_value
                any_signal = True
                trigger_found += 1
                if float(trigger_logit_value) > 0.0:
                    trigger_pos += 1
            else:
                fallback_used += 1
                row_has_all_fields = False

            delta_value = _safe_float(
                source.get("delta_pred", source.get("delta_action_norm", source.get("delta_norm")))
            )
            if delta_value is not None:
                delta_pred[i] = delta_value
                any_signal = True
                delta_found += 1
                delta_sum += float(delta_value)
            else:
                fallback_used += 1
                row_has_all_fields = False

            region_raw = source.get(
                "region_logits",
                source.get("region_prior_pred", source.get("affected_region_prior")),
            )
            if isinstance(region_raw, (list, tuple)):
                region_input_present += 1
                expected_region_len = self.expected_region_len if self.expected_region_len is not None else self.chunk_len
                if len(region_raw) != expected_region_len:
                    region_len_mismatch += 1
                    if self.strict_output_contract:
                        raise ValueError(
                            "Standalone A output contract violation: "
                            f"region len={len(region_raw)} != expected_region_len={expected_region_len}"
                        )
            region_vec = _coerce_vector(region_raw, target_len=self.chunk_len)
            if region_vec is not None:
                region_logits[i] = region_vec.to(device=device, dtype=dtype)
                any_signal = True
                region_found += 1
            else:
                fallback_used += 1
                row_has_all_fields = False

            if row_has_all_fields:
                rows_with_all_fields += 1

        denom = max(1, batch_size)
        trigger_rate_denom = max(1, trigger_found)
        region_present_denom = max(1, region_input_present)
        embedding_present_denom = max(1, embedding_present)
        debug_metrics = {
            "debug/a_module_mode_standalone": 1.0,
            "debug/a_module_external_signal_found": 1.0 if any_signal else 0.0,
            "debug/a_module_external_missing_fields": float(fallback_used),
            "debug/a_module_external_rows_with_payload_ratio": float(rows_with_payload / denom),
            "debug/a_module_external_rows_with_all_fields_ratio": float(rows_with_all_fields / denom),
            "debug/a_module_external_risk_coverage": float(risk_found / denom),
            "debug/a_module_external_trigger_coverage": float(trigger_found / denom),
            "debug/a_module_external_delta_coverage": float(delta_found / denom),
            "debug/a_module_external_region_coverage": float(region_found / denom),
            "debug/a_module_external_risk_mean": float(risk_sum / max(1, risk_found)),
            "debug/a_module_external_delta_mean": float(delta_sum / max(1, delta_found)),
            "debug/a_module_external_trigger_pos_rate": float(trigger_pos / trigger_rate_denom),
            "debug/a_module_external_region_input_present_ratio": float(region_input_present / denom),
            "debug/a_module_external_region_len_mismatch_ratio": float(region_len_mismatch / region_present_denom),
            "debug/a_module_external_embedding_present_ratio": float(embedding_present / denom),
            "debug/a_module_external_embedding_len_mismatch_ratio": float(
                embedding_len_mismatch / embedding_present_denom
            ),
        }
        return (
            AModulePredictions(
                risk_pred=risk_pred,
                trigger_logit=trigger_logit,
                delta_pred=delta_pred,
                region_logits=region_logits,
                debug_metrics=debug_metrics,
            ),
            any_signal,
        )

    def predict(
        self,
        *,
        pooled_hidden: torch.Tensor,
        examples: list[dict],
        action_loss: torch.Tensor,
        chunk_len: int,
    ) -> AModulePredictions:
        _ = pooled_hidden, chunk_len
        predictions, any_signal = self._prediction_from_examples(
            examples=examples,
            device=action_loss.device,
            dtype=action_loss.dtype,
        )
        if any_signal:
            return predictions

        if self.fallback_to_lite is not None:
            lite_predictions = self.fallback_to_lite.predict(
                pooled_hidden=pooled_hidden,
                examples=examples,
                action_loss=action_loss,
                chunk_len=chunk_len,
            )
            lite_predictions.debug_metrics.update(predictions.debug_metrics)
            lite_predictions.debug_metrics["debug/a_module_standalone_fallback_to_lite"] = 1.0
            return lite_predictions

        if self.strict_missing:
            raise KeyError(
                f"Standalone A module expects key={self.input_key!r} in examples, "
                "but no usable prediction fields were found."
            )
        return predictions


def build_a_module_interface(
    *,
    config,
    chunk_len: int,
    a_risk_head,
    a_trigger_head,
    corrective_delta_head,
    corrective_region_head,
):
    """Factory for A-module prediction interface.

    Modes:
      - lite: use in-model lightweight heads (current default behavior)
      - standalone: read external A predictions from sample payload
    """
    framework_cfg = getattr(config, "framework", None)
    a_cfg = _cfg_get(framework_cfg, "a_module", None)
    mode = str(_cfg_get(a_cfg, "mode", "lite")).strip().lower()

    lite_interface = LiteAModuleInterface(
        a_risk_head=a_risk_head,
        a_trigger_head=a_trigger_head,
        corrective_delta_head=corrective_delta_head,
        corrective_region_head=corrective_region_head,
    )
    if mode == "lite":
        return lite_interface

    if mode == "standalone":
        standalone_cfg = _cfg_get(a_cfg, "standalone", None)
        input_key = str(_cfg_get(standalone_cfg, "input_key", "a_outputs"))
        allow_pseudo_labels_fallback = bool(
            _cfg_get(standalone_cfg, "allow_pseudo_labels_fallback", False)
        )
        strict_missing = bool(_cfg_get(standalone_cfg, "strict_missing", False))
        fallback_to_lite = bool(_cfg_get(standalone_cfg, "fallback_to_lite", True))
        contract_cfg = _cfg_get(standalone_cfg, "output_contract", None)
        strict_output_contract = bool(_cfg_get(contract_cfg, "strict_shape", False))
        expected_region_len = _to_int_or_none(_cfg_get(contract_cfg, "expected_region_len", None))
        expected_embedding_dim = _to_int_or_none(_cfg_get(contract_cfg, "expected_embedding_dim", None))
        return StandaloneAModuleInterface(
            chunk_len=chunk_len,
            input_key=input_key,
            allow_pseudo_labels_fallback=allow_pseudo_labels_fallback,
            strict_missing=strict_missing,
            fallback_to_lite=lite_interface if fallback_to_lite else None,
            strict_output_contract=strict_output_contract,
            expected_region_len=expected_region_len,
            expected_embedding_dim=expected_embedding_dim,
        )

    raise ValueError(
        f"Unsupported framework.a_module.mode={mode!r}, expected one of ['lite', 'standalone']"
    )
