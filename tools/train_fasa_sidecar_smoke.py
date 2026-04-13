#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


CONTRACT_VERSION = "fasa_v1"
CONTRACT_SOURCE = "fasa/main"
REGION_LEN = 15
EMBED_DIM = 16
EMBED_HALF_DIM = EMBED_DIM // 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    cleaned: list[float] = []
    for item in value:
        scalar = _safe_float(item)
        if scalar is None:
            return None
        cleaned.append(float(scalar))
    return cleaned


def _reduce_delta(value: Any) -> float | None:
    scalar = _safe_float(value)
    if scalar is not None:
        return float(max(0.0, scalar))
    values = _finite_list(value)
    if not values:
        return None
    return float(max(0.0, max(values)))


def _pad_or_trim(values: list[float], target_len: int) -> list[float]:
    if len(values) >= target_len:
        return [float(v) for v in values[:target_len]]
    padded = list(values)
    padded.extend([0.0] * (target_len - len(padded)))
    return [float(v) for v in padded]


def _all_finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


@dataclass
class Phase0Tensors:
    features: torch.Tensor
    risk_target: torch.Tensor
    trigger_target: torch.Tensor
    delta_target: torch.Tensor
    region_target: torch.Tensor
    embed_target: torch.Tensor
    rows_total: int
    state_dim: int
    future_dim: int
    action_flat_dim: int
    horizon_steps: int


def _load_phase0(path: Path) -> Phase0Tensors:
    features: list[list[float]] = []
    risk_target: list[float] = []
    trigger_target: list[float] = []
    delta_target: list[float] = []
    region_target: list[list[float]] = []
    embed_target: list[list[float]] = []
    state_dim = 0
    future_dim = 0
    action_flat_dim = 0
    horizon_steps = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            state_t = _finite_list(record.get("state_t"))
            future_state = _finite_list(record.get("future_state"))
            region = _finite_list(record.get("region_target_15"))
            action_chunk = record.get("action_chunk")
            pseudo = record.get("pseudo_labels")
            if (
                state_t is None
                or future_state is None
                or region is None
                or not isinstance(action_chunk, list)
                or not isinstance(pseudo, dict)
            ):
                raise ValueError(f"Invalid Phase0 row at line {line_no}")
            if len(region) != REGION_LEN:
                raise ValueError(
                    f"region_target_15 len={len(region)} != {REGION_LEN} at line {line_no}"
                )

            flat_action: list[float] = []
            for row in action_chunk:
                row_values = _finite_list(row)
                if row_values is None:
                    raise ValueError(f"action_chunk contains non-finite values at line {line_no}")
                flat_action.extend(row_values)

            risk_value = _safe_float(pseudo.get("risk_score"))
            trigger_value = _safe_float(pseudo.get("trigger_label"))
            delta_value = _reduce_delta(pseudo.get("delta_action_norm"))
            if risk_value is None or trigger_value is None or delta_value is None:
                raise ValueError(f"pseudo_labels missing finite targets at line {line_no}")

            risk_value = float(max(0.0, min(1.0, risk_value)))
            trigger_value = 1.0 if trigger_value >= 0.5 else 0.0
            delta_value = float(max(0.0, delta_value))
            delta_state = [float(f - s) for s, f in zip(state_t, future_state)]
            feature = list(state_t) + list(future_state) + list(delta_state) + flat_action

            embed = _pad_or_trim(state_t, EMBED_HALF_DIM) + _pad_or_trim(future_state, EMBED_HALF_DIM)

            features.append(feature)
            risk_target.append(risk_value)
            trigger_target.append(trigger_value)
            delta_target.append(delta_value)
            region_target.append(region)
            embed_target.append(embed)
            state_dim = max(state_dim, len(state_t))
            future_dim = max(future_dim, len(future_state))
            action_flat_dim = max(action_flat_dim, len(flat_action))
            horizon_steps = max(horizon_steps, int(record.get("horizon_steps", 0)))

    if not features:
        raise RuntimeError(f"No rows loaded from {path}")

    feature_dim = len(features[0])
    if any(len(row) != feature_dim for row in features):
        raise ValueError("Feature dimension mismatch across Phase0 rows")

    tensors = Phase0Tensors(
        features=torch.tensor(features, dtype=torch.float32),
        risk_target=torch.tensor(risk_target, dtype=torch.float32),
        trigger_target=torch.tensor(trigger_target, dtype=torch.float32),
        delta_target=torch.tensor(delta_target, dtype=torch.float32),
        region_target=torch.tensor(region_target, dtype=torch.float32),
        embed_target=torch.tensor(embed_target, dtype=torch.float32),
        rows_total=len(features),
        state_dim=state_dim,
        future_dim=future_dim,
        action_flat_dim=action_flat_dim,
        horizon_steps=horizon_steps,
    )
    if not all(
        _all_finite_tensor(tensor)
        for tensor in (
            tensors.features,
            tensors.risk_target,
            tensors.trigger_target,
            tensors.delta_target,
            tensors.region_target,
            tensors.embed_target,
        )
    ):
        raise ValueError("Loaded Phase0 tensors contain non-finite values")
    return tensors


class FASASidecarMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.trigger_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.region_head = nn.Linear(hidden_dim, REGION_LEN)
        self.embed_head = nn.Linear(hidden_dim, EMBED_DIM)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        risk_logit = self.risk_head(hidden).squeeze(-1)
        trigger_logit = self.trigger_head(hidden).squeeze(-1)
        delta_raw = self.delta_head(hidden).squeeze(-1)
        region_logits = self.region_head(hidden)
        dynamic_embedding = self.embed_head(hidden)
        return {
            "risk_pred": torch.sigmoid(risk_logit),
            "trigger_logit": trigger_logit,
            "delta_pred": F.softplus(delta_raw),
            "region_logits": region_logits,
            "dynamic_embedding": dynamic_embedding,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standalone FASA sidecar smoke model.")
    parser.add_argument("--input-jsonl", required=True, help="Archived Phase0 JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Run artifact directory.")
    parser.add_argument("--max-steps", type=int, required=True, help="Train steps for this smoke run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Training device.",
    )
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    requested = str(name).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    return torch.device(requested)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message, flush=True)


def _build_config(args: argparse.Namespace, tensors: Phase0Tensors, run_id: str, device: torch.device) -> dict[str, Any]:
    return {
        "run": {
            "run_id": run_id,
            "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        },
        "data": {
            "phase0_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
            "rows_total": tensors.rows_total,
            "state_dim": tensors.state_dim,
            "future_dim": tensors.future_dim,
            "action_flat_dim": tensors.action_flat_dim,
            "horizon_steps": tensors.horizon_steps,
        },
        "trainer": {
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "device": str(device),
        },
        "model": {
            "hidden_dim": int(args.hidden_dim),
            "input_dim": int(tensors.features.shape[1]),
        },
        "contract": {
            "risk_pred": "scalar[0,1]",
            "trigger_logit": "scalar_logit",
            "delta_pred": "scalar_non_negative",
            "region_logits_len": REGION_LEN,
            "dynamic_embedding_len": EMBED_DIM,
            "version": CONTRACT_VERSION,
            "source": CONTRACT_SOURCE,
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Phase0 JSONL not found: {input_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = output_dir.name
    log_path = output_dir / "train.raw.log"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.jsonl"
    config_path = output_dir / "config.yaml"
    contract_path = output_dir / "contract_shape_check.json"
    identity_path = output_dir / "run_identity.txt"

    if log_path.exists():
        log_path.unlink()
    if metrics_path.exists():
        metrics_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    start_time = _utc_now()

    tensors = _load_phase0(input_path)
    config = _build_config(args, tensors, run_id, device)
    _write_text(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    model = FASASidecarMLP(input_dim=tensors.features.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    _append_log(log_path, f"begin run_id={run_id} steps={args.max_steps} device={device.type}")
    _append_log(log_path, f"rows={tensors.rows_total} feature_dim={tensors.features.shape[1]}")

    for step in range(1, int(args.max_steps) + 1):
        batch_idx = torch.randint(0, tensors.rows_total, (int(args.batch_size),))
        batch_features = tensors.features[batch_idx].to(device)
        batch_risk = tensors.risk_target[batch_idx].to(device)
        batch_trigger = tensors.trigger_target[batch_idx].to(device)
        batch_delta = tensors.delta_target[batch_idx].to(device)
        batch_region = tensors.region_target[batch_idx].to(device)
        batch_embed = tensors.embed_target[batch_idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_features)
        loss_risk = F.mse_loss(outputs["risk_pred"], batch_risk)
        loss_trigger = F.binary_cross_entropy_with_logits(outputs["trigger_logit"], batch_trigger)
        loss_delta = F.mse_loss(outputs["delta_pred"], batch_delta)
        loss_region = F.binary_cross_entropy_with_logits(outputs["region_logits"], batch_region)
        loss_embed = F.mse_loss(outputs["dynamic_embedding"], batch_embed)
        loss_total = loss_risk + loss_trigger + loss_delta + loss_region + loss_embed
        if not _all_finite_tensor(loss_total.detach()):
            raise RuntimeError(f"non-finite total loss at step {step}")
        loss_total.backward()
        optimizer.step()

        metrics = {
            "step": step,
            "loss/risk": float(loss_risk.detach().cpu().item()),
            "loss/trigger": float(loss_trigger.detach().cpu().item()),
            "loss/delta": float(loss_delta.detach().cpu().item()),
            "loss/region": float(loss_region.detach().cpu().item()),
            "loss/embed": float(loss_embed.detach().cpu().item()),
            "loss/total": float(loss_total.detach().cpu().item()),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        _append_log(
            log_path,
            "step="
            f"{step} "
            f"loss/total={metrics['loss/total']:.6f} "
            f"loss/risk={metrics['loss/risk']:.6f} "
            f"loss/trigger={metrics['loss/trigger']:.6f} "
            f"loss/delta={metrics['loss/delta']:.6f} "
            f"loss/region={metrics['loss/region']:.6f} "
            f"loss/embed={metrics['loss/embed']:.6f}",
        )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"steps": int(args.max_steps)}, ensure_ascii=False) + "\n")

    model.eval()
    with torch.no_grad():
        sample_outputs = model(tensors.features[:1].to(device))
    payload = {
        "risk_pred": float(sample_outputs["risk_pred"][0].detach().cpu().item()),
        "trigger_logit": float(sample_outputs["trigger_logit"][0].detach().cpu().item()),
        "delta_pred": float(sample_outputs["delta_pred"][0].detach().cpu().item()),
        "region_logits": [float(v) for v in sample_outputs["region_logits"][0].detach().cpu().tolist()],
        "dynamic_embedding": [float(v) for v in sample_outputs["dynamic_embedding"][0].detach().cpu().tolist()],
        "version": CONTRACT_VERSION,
        "source": CONTRACT_SOURCE,
    }
    contract_check = {
        "contract_ok": (
            len(payload["region_logits"]) == REGION_LEN
            and len(payload["dynamic_embedding"]) == EMBED_DIM
            and math.isfinite(payload["risk_pred"])
            and math.isfinite(payload["trigger_logit"])
            and math.isfinite(payload["delta_pred"])
            and all(math.isfinite(v) for v in payload["region_logits"])
            and all(math.isfinite(v) for v in payload["dynamic_embedding"])
        ),
        "keys": list(payload.keys()),
        "region_logits_len": len(payload["region_logits"]),
        "dynamic_embedding_len": len(payload["dynamic_embedding"]),
        "version": payload["version"],
        "source": payload["source"],
        "sample_payload": payload,
    }
    _write_text(contract_path, json.dumps(contract_check, ensure_ascii=False, indent=2) + "\n")
    if not contract_check["contract_ok"]:
        raise RuntimeError("contract shape check failed")

    end_time = _utc_now()
    identity_lines = [
        f"run_id={run_id}",
        f"script={Path(__file__).resolve()}",
        f"input_jsonl={input_path}",
        f"output_dir={output_dir}",
        f"max_steps={args.max_steps}",
        f"batch_size={args.batch_size}",
        f"lr={args.lr}",
        f"seed={args.seed}",
        f"device={device.type}",
        f"started_at={start_time}",
        f"ended_at={end_time}",
        "exit_status=0",
    ]
    _write_text(identity_path, "\n".join(identity_lines) + "\n")
    _append_log(log_path, f"done steps={args.max_steps}")


if __name__ == "__main__":
    main()
