from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
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


_VERSIONED_DATASET_RE = re.compile(r"^(?P<prefix>.+?)_\d+\.\d+\.\d+_lerobot$")


def _dataset_name_aliases(dataset_name: str) -> list[str]:
    raw = str(dataset_name)
    aliases = {raw}
    basename = Path(raw).name
    aliases.add(basename)

    match = _VERSIONED_DATASET_RE.match(basename)
    if match is not None:
        aliases.add(match.group("prefix"))
    if basename.endswith("_lerobot"):
        aliases.add(basename[: -len("_lerobot")])
    return [alias for alias in aliases if alias]


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


def _extract_runtime_payload(record: dict, input_key: str) -> dict | None:
    payload = record.get(input_key)
    if isinstance(payload, dict):
        return payload
    pseudo = record.get("pseudo_labels")
    if isinstance(pseudo, dict):
        pseudo_payload = pseudo.get(input_key)
        if isinstance(pseudo_payload, dict):
            return pseudo_payload
    required_keys = {"risk_pred", "trigger_logit", "delta_pred", "region_logits", "dynamic_embedding"}
    if required_keys.issubset(record.keys()):
        return {key: record.get(key) for key in required_keys.union({"version", "source"})}
    return None


def _load_runtime_a_outputs_index(jsonl_path: str, input_key: str) -> tuple[dict[tuple[str, int, int], dict], dict[str, int]]:
    path = Path(jsonl_path).expanduser()
    stats = {
        "rows": 0,
        "indexed": 0,
        "duplicates": 0,
        "invalid": 0,
        "invalid_json": 0,
        "invalid_meta": 0,
        "invalid_payload": 0,
    }
    if not path.exists():
        raise FileNotFoundError(f"in-loop runtime jsonl not found: {path}")

    index: dict[tuple[str, int, int], dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                stats["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                stats["invalid"] += 1
                stats["invalid_json"] += 1
                continue

            meta = record.get("meta")
            if not isinstance(meta, dict):
                stats["invalid"] += 1
                stats["invalid_meta"] += 1
                continue
            dataset_name = str(meta.get("dataset_name", "")).strip()
            trajectory_id = _safe_int(meta.get("trajectory_id"))
            sample_step = _safe_int(meta.get("sample_step"))
            if not dataset_name or trajectory_id is None or sample_step is None:
                stats["invalid"] += 1
                stats["invalid_meta"] += 1
                continue

            payload = _extract_runtime_payload(record, input_key=input_key)
            if not isinstance(payload, dict):
                stats["invalid"] += 1
                stats["invalid_payload"] += 1
                continue

            for alias in _dataset_name_aliases(dataset_name):
                key = (alias, trajectory_id, sample_step)
                if key in index:
                    stats["duplicates"] += 1
                index[key] = payload
                stats["indexed"] += 1

    return index, stats


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
        vdpm_mode: str = "offline",
        vdpm_source: str = "precomputed",
        vdpm_runtime_jsonl: str | None = None,
    ) -> None:
        self.chunk_len = int(chunk_len)
        self.input_key = str(input_key)
        self.allow_pseudo_labels_fallback = bool(allow_pseudo_labels_fallback)
        self.strict_missing = bool(strict_missing)
        self.fallback_to_lite = fallback_to_lite
        self.strict_output_contract = bool(strict_output_contract)
        self.expected_region_len = int(expected_region_len) if expected_region_len else None
        self.expected_embedding_dim = int(expected_embedding_dim) if expected_embedding_dim else None
        self.vdpm_mode = str(vdpm_mode or "offline").strip().lower()
        self.vdpm_source = str(vdpm_source or "precomputed").strip().lower()
        self.vdpm_runtime_jsonl = str(vdpm_runtime_jsonl or "").strip()
        self.inloop_enabled = self.vdpm_mode == "inloop"
        self.inloop_runtime_enabled = self.inloop_enabled and self.vdpm_source == "runtime_infer"

        self._inloop_calls = 0
        self._inloop_success = 0
        self._inloop_fail = 0
        self._inloop_fallback_count = 0
        self._inloop_latency_ms: list[float] = []
        self._inloop_runtime_index: dict[tuple[str, int, int], dict] = {}
        self._inloop_runtime_index_loaded = False
        self._inloop_runtime_index_stats = {
            "rows": 0,
            "indexed": 0,
            "duplicates": 0,
            "invalid": 0,
            "invalid_json": 0,
            "invalid_meta": 0,
            "invalid_payload": 0,
        }
        self._inloop_runtime_load_error = ""

        if self.inloop_runtime_enabled and self.vdpm_runtime_jsonl:
            try:
                index, stats = _load_runtime_a_outputs_index(
                    self.vdpm_runtime_jsonl,
                    input_key=self.input_key,
                )
                self._inloop_runtime_index = index
                self._inloop_runtime_index_stats = stats
                self._inloop_runtime_index_loaded = True
            except Exception as exc:
                self._inloop_runtime_load_error = str(exc)

    def _extract_precomputed_source(self, example: dict) -> dict:
        source = example.get(self.input_key)
        if isinstance(source, dict):
            return source
        # Optional fallback for migration. Keep disabled by default to avoid label leakage.
        pseudo = example.get("pseudo_labels")
        if self.allow_pseudo_labels_fallback and isinstance(pseudo, dict):
            return pseudo
        return {}

    def _lookup_runtime_source(self, example: dict) -> dict | None:
        if not self.inloop_runtime_enabled:
            return None
        if not self._inloop_runtime_index_loaded:
            return None
        meta = example.get("meta")
        if not isinstance(meta, dict):
            return None
        dataset_name = str(meta.get("dataset_name", "")).strip()
        trajectory_id = _safe_int(meta.get("trajectory_id"))
        sample_step = _safe_int(meta.get("sample_step"))
        if not dataset_name or trajectory_id is None or sample_step is None:
            return None
        for alias in _dataset_name_aliases(dataset_name):
            key = (alias, trajectory_id, sample_step)
            payload = self._inloop_runtime_index.get(key)
            if isinstance(payload, dict):
                return payload
        return None

    def _record_inloop_call(self, elapsed_ms: float, success: bool, fallback: bool) -> None:
        self._inloop_calls += 1
        if success:
            self._inloop_success += 1
        else:
            self._inloop_fail += 1
        if fallback:
            self._inloop_fallback_count += 1
        if math.isfinite(float(elapsed_ms)) and elapsed_ms >= 0.0:
            self._inloop_latency_ms.append(float(elapsed_ms))
            if len(self._inloop_latency_ms) > 4096:
                self._inloop_latency_ms = self._inloop_latency_ms[-4096:]

    def _inloop_latency_stats(self) -> tuple[float, float]:
        if not self._inloop_latency_ms:
            return 0.0, 0.0
        values = sorted(self._inloop_latency_ms)
        mean = sum(values) / len(values)
        p95_idx = max(0, min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1))
        return float(mean), float(values[p95_idx])

    def _extract_source(self, example: dict) -> dict:
        if not self.inloop_enabled:
            return self._extract_precomputed_source(example)

        started = time.perf_counter()
        success = False
        fallback = False

        source = {}
        if self.inloop_runtime_enabled:
            runtime_source = self._lookup_runtime_source(example)
            if isinstance(runtime_source, dict) and runtime_source:
                source = runtime_source
                example[self.input_key] = runtime_source
                success = True
            else:
                source = self._extract_precomputed_source(example)
                fallback = bool(source)
        else:
            source = self._extract_precomputed_source(example)
            success = bool(source)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not success and not fallback and source:
            success = True
        self._record_inloop_call(elapsed_ms=elapsed_ms, success=success, fallback=fallback)
        return source

    def _inloop_debug_metrics(self) -> dict[str, float]:
        latency_mean, latency_p95 = self._inloop_latency_stats()
        calls_denom = max(1, self._inloop_calls)
        return {
            "inloop_calls": float(self._inloop_calls),
            "inloop_success": float(self._inloop_success),
            "inloop_fail": float(self._inloop_fail),
            "inloop_fallback_count": float(self._inloop_fallback_count),
            "inloop_infer_success_ratio": float(self._inloop_success / calls_denom),
            "inloop_latency_ms_mean": float(latency_mean),
            "inloop_latency_ms_p95": float(latency_p95),
            "debug/inloop_mode_enabled": 1.0 if self.inloop_enabled else 0.0,
            "debug/inloop_runtime_source_enabled": 1.0 if self.inloop_runtime_enabled else 0.0,
            "debug/inloop_runtime_index_loaded": 1.0 if self._inloop_runtime_index_loaded else 0.0,
            "debug/inloop_runtime_index_rows": float(self._inloop_runtime_index_stats.get("indexed", 0)),
            "debug/inloop_runtime_index_invalid_rows": float(self._inloop_runtime_index_stats.get("invalid", 0)),
            "debug/inloop_runtime_load_error": 1.0 if self._inloop_runtime_load_error else 0.0,
        }

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
        debug_metrics.update(self._inloop_debug_metrics())
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
        vdpm_mode = str(_cfg_get(a_cfg, "vdpm_mode", "offline")).strip().lower()
        vdpm_source = str(_cfg_get(a_cfg, "vdpm_source", "precomputed")).strip().lower()
        vdpm_runtime_jsonl = _cfg_get(
            a_cfg,
            "vdpm_runtime_jsonl",
            _cfg_get(standalone_cfg, "vdpm_runtime_jsonl", None),
        )
        return StandaloneAModuleInterface(
            chunk_len=chunk_len,
            input_key=input_key,
            allow_pseudo_labels_fallback=allow_pseudo_labels_fallback,
            strict_missing=strict_missing,
            fallback_to_lite=lite_interface if fallback_to_lite else None,
            strict_output_contract=strict_output_contract,
            expected_region_len=expected_region_len,
            expected_embedding_dim=expected_embedding_dim,
            vdpm_mode=vdpm_mode,
            vdpm_source=vdpm_source,
            vdpm_runtime_jsonl=vdpm_runtime_jsonl,
        )

    raise ValueError(
        f"Unsupported framework.a_module.mode={mode!r}, expected one of ['lite', 'standalone']"
    )
