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

import numpy as np

REGION_LEN = 15
EMBED_DIM = 16
OUT_DIM = 3 + REGION_LEN + EMBED_DIM
CONTRACT_KEYS = {
    "risk_pred",
    "trigger_logit",
    "delta_pred",
    "region_logits",
    "dynamic_embedding",
    "version",
    "source",
}


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
    out: list[float] = []
    for item in value:
        f = _safe_float(item)
        if f is None:
            return None
        out.append(float(f))
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _append_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text, flush=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataclass
class AOutputsTensors:
    features: np.ndarray  # [N, D]
    risk_target: np.ndarray  # [N]
    trigger_target: np.ndarray  # [N]
    delta_target: np.ndarray  # [N]
    region_target: np.ndarray  # [N, 15]
    embed_target: np.ndarray  # [N, 16]
    rows_total: int


def _load_a_outputs_jsonl(path: Path) -> AOutputsTensors:
    features: list[list[float]] = []
    risk_target: list[float] = []
    trigger_target: list[float] = []
    delta_target: list[float] = []
    region_target: list[list[float]] = []
    embed_target: list[list[float]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_no}: row is not an object")
            payload = obj.get("a_outputs")
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_no}: missing a_outputs")
            if not CONTRACT_KEYS.issubset(set(payload.keys())):
                missing = sorted(CONTRACT_KEYS - set(payload.keys()))
                raise ValueError(f"line {line_no}: missing keys {missing}")

            risk = _safe_float(payload.get("risk_pred"))
            trigger = _safe_float(payload.get("trigger_logit"))
            delta = _safe_float(payload.get("delta_pred"))
            region = _finite_list(payload.get("region_logits"))
            embed = _finite_list(payload.get("dynamic_embedding"))
            horizon = _safe_float(obj.get("horizon_steps"))

            if risk is None or trigger is None or delta is None:
                raise ValueError(f"line {line_no}: scalar fields not finite")
            if delta < 0.0:
                raise ValueError(f"line {line_no}: delta_pred negative")
            if region is None or len(region) != REGION_LEN:
                raise ValueError(f"line {line_no}: region_logits len mismatch")
            if embed is None or len(embed) != EMBED_DIM:
                raise ValueError(f"line {line_no}: dynamic_embedding len mismatch")
            if not isinstance(payload.get("version"), str) or not payload.get("version").strip():
                raise ValueError(f"line {line_no}: version missing")
            if not isinstance(payload.get("source"), str) or not payload.get("source").strip():
                raise ValueError(f"line {line_no}: source missing")

            feature = list(region) + list(embed) + [float(horizon if horizon is not None else 0.0)]

            features.append(feature)
            risk_target.append(float(max(0.0, min(1.0, risk))))
            trigger_target.append(float(trigger))
            delta_target.append(float(delta))
            region_target.append([float(v) for v in region])
            embed_target.append([float(v) for v in embed])

    if not features:
        raise RuntimeError(f"no rows loaded from {path}")

    feat = np.asarray(features, dtype=np.float32)
    risk = np.asarray(risk_target, dtype=np.float32)
    trig = np.asarray(trigger_target, dtype=np.float32)
    delt = np.asarray(delta_target, dtype=np.float32)
    reg = np.asarray(region_target, dtype=np.float32)
    emb = np.asarray(embed_target, dtype=np.float32)

    all_arr = [feat, risk, trig, delt, reg, emb]
    if not all(np.isfinite(arr).all() for arr in all_arr):
        raise RuntimeError("loaded arrays contain non-finite values")

    return AOutputsTensors(
        features=feat,
        risk_target=risk,
        trigger_target=trig,
        delta_target=delt,
        region_target=reg,
        embed_target=emb,
        rows_total=int(feat.shape[0]),
    )


def _metrics_key_summary(metrics_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("metrics is empty")

    keys = ["loss/total", "loss/risk", "loss/trigger", "loss/delta", "loss/region", "loss/embed"]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "all_loss_finite": True,
        "initial": {},
        "final": {},
        "min": {},
    }
    for key in keys:
        vals = [_safe_float(row.get(key)) for row in rows]
        if any(v is None for v in vals):
            summary["all_loss_finite"] = False
            continue
        valsf = [float(v) for v in vals if v is not None]
        if not all(math.isfinite(v) for v in valsf):
            summary["all_loss_finite"] = False
        summary["initial"][key] = valsf[0]
        summary["final"][key] = valsf[-1]
        summary["min"][key] = min(valsf)
    summary["final_step"] = int(rows[-1].get("step", len(rows)))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase1.3 sidecar-only A-module train check on a_outputs JSONL")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input jsonl not found: {input_path}")

    run_id = output_dir.name
    log_path = output_dir / "train.raw.log"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.jsonl"
    config_path = output_dir / "config.json"
    summary_key_path = output_dir / "metrics_key_summary.json"
    identity_path = output_dir / "run_identity.txt"

    for p in (log_path, metrics_path, summary_path):
        if p.exists():
            p.unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tensors = _load_a_outputs_jsonl(input_path)
    n_rows, feat_dim = tensors.features.shape

    config = {
        "run": {"run_id": run_id, "output_dir": str(output_dir)},
        "data": {"input_jsonl": str(input_path), "rows_total": tensors.rows_total, "feature_dim": feat_dim},
        "trainer": {
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "backend": "numpy_linear",
        },
        "model": {"input_dim": feat_dim, "output_dim": OUT_DIM},
    }
    _write_text(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    # linear model params
    w = np.random.normal(0.0, 0.01, size=(feat_dim, OUT_DIM)).astype(np.float32)
    b = np.zeros((OUT_DIM,), dtype=np.float32)

    started = _utc_now()
    _append_log(log_path, f"begin run_id={run_id} steps={args.max_steps} backend=numpy_linear")
    _append_log(log_path, f"rows={n_rows} feature_dim={feat_dim}")

    for step in range(1, int(args.max_steps) + 1):
        idx = np.random.randint(0, n_rows, size=(int(args.batch_size),))
        x = tensors.features[idx]  # [B, D]
        risk_t = tensors.risk_target[idx]  # [B]
        trigger_t = tensors.trigger_target[idx]
        delta_t = tensors.delta_target[idx]
        region_t = tensors.region_target[idx]  # [B, 15]
        embed_t = tensors.embed_target[idx]  # [B, 16]

        y = x @ w + b  # [B, OUT_DIM]
        risk_logit = y[:, 0]
        trigger_p = y[:, 1]
        delta_raw = y[:, 2]
        region_p = y[:, 3 : 3 + REGION_LEN]
        embed_p = y[:, 3 + REGION_LEN :]

        risk_pred = _sigmoid(risk_logit)
        delta_pred = _softplus(delta_raw)

        bsz = float(x.shape[0])
        loss_risk = float(np.mean((risk_pred - risk_t) ** 2))
        loss_trigger = float(np.mean((trigger_p - trigger_t) ** 2))
        loss_delta = float(np.mean((delta_pred - delta_t) ** 2))
        loss_region = float(np.mean((region_p - region_t) ** 2))
        loss_embed = float(np.mean((embed_p - embed_t) ** 2))
        loss_total = float(loss_risk + loss_trigger + loss_delta + loss_region + loss_embed)

        losses = [loss_total, loss_risk, loss_trigger, loss_delta, loss_region, loss_embed]
        if not all(math.isfinite(v) for v in losses):
            raise RuntimeError(f"non-finite loss at step {step}")

        # grads dL/dy
        grad = np.zeros_like(y, dtype=np.float32)
        grad[:, 0] = (2.0 / bsz) * (risk_pred - risk_t) * (risk_pred * (1.0 - risk_pred))
        grad[:, 1] = (2.0 / bsz) * (trigger_p - trigger_t)
        grad[:, 2] = (2.0 / bsz) * (delta_pred - delta_t) * _sigmoid(delta_raw)
        grad[:, 3 : 3 + REGION_LEN] = (2.0 / (bsz * REGION_LEN)) * (region_p - region_t)
        grad[:, 3 + REGION_LEN :] = (2.0 / (bsz * EMBED_DIM)) * (embed_p - embed_t)

        grad_w = x.T @ grad
        grad_b = np.sum(grad, axis=0)

        w -= float(args.lr) * grad_w
        b -= float(args.lr) * grad_b

        metrics = {
            "step": step,
            "loss/risk": loss_risk,
            "loss/trigger": loss_trigger,
            "loss/delta": loss_delta,
            "loss/region": loss_region,
            "loss/embed": loss_embed,
            "loss/total": loss_total,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        _append_log(
            log_path,
            f"step={step} loss/total={loss_total:.6f} "
            f"loss/risk={loss_risk:.6f} "
            f"loss/trigger={loss_trigger:.6f} "
            f"loss/delta={loss_delta:.6f} "
            f"loss/region={loss_region:.6f} "
            f"loss/embed={loss_embed:.6f}",
        )

    _write_text(summary_path, json.dumps({"steps": int(args.max_steps)}, ensure_ascii=False) + "\n")

    key_summary = _metrics_key_summary(metrics_path)
    _write_text(summary_key_path, json.dumps(key_summary, ensure_ascii=False, indent=2) + "\n")
    if not key_summary.get("all_loss_finite", False):
        raise RuntimeError("metrics_key_summary indicates non-finite loss")

    ended = _utc_now()
    identity = [
        f"run_id={run_id}",
        f"script={Path(__file__).resolve()}",
        f"input_jsonl={input_path}",
        f"output_dir={output_dir}",
        f"max_steps={args.max_steps}",
        f"batch_size={args.batch_size}",
        f"lr={args.lr}",
        f"seed={args.seed}",
        "backend=numpy_linear",
        f"started_at={started}",
        f"ended_at={ended}",
        "exit_status=0",
    ]
    _write_text(identity_path, "\n".join(identity) + "\n")
    _append_log(log_path, f"done steps={args.max_steps}")


if __name__ == "__main__":
    main()
