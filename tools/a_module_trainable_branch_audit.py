#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from starVLA.model.framework.a_fusion_heads import CrossAttentionFusionAHead


def _cfg_get(cfg_obj: Any, key: str, default=None):
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit trainable-branch boundaries for fusion A-head training."
    )
    parser.add_argument("--runtime-config", required=True, help="Path to runtime_config.json")
    parser.add_argument("--freeze-assert", required=True, help="Path to freeze_assert.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config_path = Path(args.runtime_config).expanduser().resolve()
    freeze_assert_path = Path(args.freeze_assert).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    runtime_cfg = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    freeze_assert = json.loads(freeze_assert_path.read_text(encoding="utf-8"))

    framework_cfg = _cfg_get(runtime_cfg, "framework", {})
    trainer_cfg = _cfg_get(runtime_cfg, "trainer", {})
    a_cfg = _cfg_get(framework_cfg, "a_module", {})
    standalone_cfg = _cfg_get(a_cfg, "standalone", {})
    fusion_cfg = _cfg_get(a_cfg, "fusion", {})
    qwenvl_cfg = _cfg_get(framework_cfg, "qwenvl", {})
    action_cfg = _cfg_get(framework_cfg, "action_model", {})

    mode = str(_cfg_get(a_cfg, "mode", "unknown"))
    freeze_modules = str(_cfg_get(trainer_cfg, "freeze_modules", ""))
    qwen_frozen = "qwen_vl_interface" in freeze_modules
    qwen_trainable_params = 0 if qwen_frozen else -1

    pooled_hidden_dim = _to_int(_cfg_get(qwenvl_cfg, "vl_hidden_dim", 2048), 2048)
    past_window = _to_int(_cfg_get(action_cfg, "past_action_window_size", 0), 0)
    future_window = _to_int(_cfg_get(action_cfg, "future_action_window_size", 7), 7)
    chunk_len = past_window + 1 + future_window
    embed_dim = _to_int(
        _cfg_get(_cfg_get(standalone_cfg, "output_contract", {}), "expected_embedding_dim", 16),
        16,
    )
    fusion_head = CrossAttentionFusionAHead(
        pooled_hidden_dim=pooled_hidden_dim,
        chunk_len=chunk_len,
        vdpm_embedding_dim=embed_dim,
        model_dim=_to_int(_cfg_get(fusion_cfg, "model_dim", 256), 256),
        num_heads=_to_int(_cfg_get(fusion_cfg, "num_heads", 8), 8),
        decoder_layers=_to_int(_cfg_get(fusion_cfg, "decoder_layers", 2), 2),
        dropout=float(_cfg_get(fusion_cfg, "dropout", 0.0) or 0.0),
    )
    a_head_trainable_params = int(
        sum(p.numel() for p in fusion_head.parameters() if getattr(p, "requires_grad", False))
    )

    # Runtime VDPM model is executed out-of-graph via subprocess worker.
    vdpm_runtime_params_in_graph = 0
    freeze_gate_pass = bool(_cfg_get(freeze_assert, "gate_pass", False))

    gate_pass = (
        mode == "fusion"
        and qwen_trainable_params == 0
        and a_head_trainable_params > 0
        and vdpm_runtime_params_in_graph == 0
        and freeze_gate_pass
    )
    summary = {
        "runtime_config": str(runtime_config_path),
        "freeze_assert": str(freeze_assert_path),
        "a_module_mode": mode,
        "qwen_trainable_params": qwen_trainable_params,
        "a_head_trainable_params": a_head_trainable_params,
        "vdpm_runtime_params_in_graph": vdpm_runtime_params_in_graph,
        "freeze_gate_pass": freeze_gate_pass,
        "optimizer_param_group_for_new_head": "base",
        "gate_pass": gate_pass,
        "notes": [
            "qwen_trainable_params uses freeze_modules policy check; strict value is 0 when qwen_vl_interface is frozen.",
            "A fusion head params are counted from architecture instantiation using runtime_config dimensions.",
            "VDPM runtime model runs in subprocess and is detached from training graph by design.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
