# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""
import contextlib
import os
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

ACTION_PATH_DTYPE = torch.float32


def _runtime_autocast_context(owner, default_dtype=torch.bfloat16):
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    runtime_mode = getattr(owner, "_starvla_runtime_mixed_precision", None)
    if runtime_mode is None:
        return torch.autocast("cuda", dtype=default_dtype)
    runtime_mode = str(runtime_mode).strip().lower()
    if runtime_mode in {"fp16", "float16", "16"}:
        return torch.autocast("cuda", dtype=torch.float16)
    if runtime_mode in {"bf16", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False)

def _cfg_get(cfg_obj, key: str, default=None):
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, "get"):
        try:
            return cfg_obj.get(key, default)
        except Exception:
            pass
    return getattr(cfg_obj, key, default)


def _cfg_enabled(cfg_obj, key: str, default: bool = False) -> bool:
    value = _cfg_get(cfg_obj, key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.a_module_interface import build_a_module_interface
from starVLA.model.framework.optional_loss_utils import build_optional_hook_targets
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################

@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise cross DiT diffusion head 
      

    Focus: Predict future continuous actions conditioned on images + instruction.
    """
# 
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # dynamic get llm config
        num_vl_layers, llm_hidden_size = 36, self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        aux_hidden_dim = max(128, llm_hidden_size // 4)
        self.a_risk_head = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, aux_hidden_dim),
            nn.SiLU(),
            nn.Linear(aux_hidden_dim, 1),
        )
        self.a_trigger_head = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, aux_hidden_dim),
            nn.SiLU(),
            nn.Linear(aux_hidden_dim, 1),
        )
        self.corrective_delta_head = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, aux_hidden_dim),
            nn.SiLU(),
            nn.Linear(aux_hidden_dim, 1),
        )
        self.corrective_region_head = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, aux_hidden_dim),
            nn.SiLU(),
            nn.Linear(aux_hidden_dim, self.chunk_len),
        )
        a_module_cfg = _cfg_get(
            _cfg_get(getattr(self.config, "framework", None), "a_module", None),
            "standalone",
            None,
        )
        output_contract_cfg = _cfg_get(a_module_cfg, "output_contract", None)
        self.a_embed_dim = int(_cfg_get(output_contract_cfg, "expected_embedding_dim", 16) or 16)
        if self.a_embed_dim <= 0:
            self.a_embed_dim = 16
        self.a_embed_head = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, aux_hidden_dim),
            nn.SiLU(),
            nn.Linear(aux_hidden_dim, self.a_embed_dim),
        )
        self.a_module_interface = build_a_module_interface(
            config=self.config,
            chunk_len=self.chunk_len,
            a_risk_head=self.a_risk_head,
            a_trigger_head=self.a_trigger_head,
            corrective_delta_head=self.corrective_delta_head,
            corrective_region_head=self.corrective_region_head,
        )

        corr_flow_cfg = _cfg_get(
            getattr(self.config, "framework", None), "corrective_flow", None
        )
        self.corrective_flow_enabled = _cfg_enabled(corr_flow_cfg, "enabled", default=False)
        if self.corrective_flow_enabled:
            from starVLA.model.framework.corrective_flow_head import CorrectiveFlowHead

            corr_hidden = int(_cfg_get(corr_flow_cfg, "hidden_dim", 1024))
            corr_layers = int(_cfg_get(corr_flow_cfg, "num_layers", 4))
            corr_heads = int(_cfg_get(corr_flow_cfg, "num_heads", 16))
            corr_region_w = float(_cfg_get(corr_flow_cfg, "region_loss_weight", 0.5))
            self.corrective_flow_head = CorrectiveFlowHead(
                action_dim=config.framework.action_model.action_dim,
                chunk_len=self.chunk_len,
                hidden_dim=corr_hidden,
                vl_hidden_dim=llm_hidden_size,
                num_layers=corr_layers,
                num_heads=corr_heads,
                region_loss_weight=corr_region_w,
            )
            _cfg_thresh = float(_cfg_get(corr_flow_cfg, "trigger_threshold", 0.5))
            _env_thresh = os.environ.get("STARVLA_CF_TRIGGER_THRESHOLD")
            self.corrective_flow_trigger_threshold = float(_env_thresh) if _env_thresh else _cfg_thresh
            if _env_thresh:
                logger.info("CF trigger_threshold overridden by env: %s (config was %s)", _env_thresh, _cfg_thresh)
            self._cf_prev_chunk: Optional[torch.Tensor] = None

    def reset_cf_state(self):
        """Reset corrective-flow episode state (call at episode boundary)."""
        self._cf_prev_chunk = None

    def _compute_optional_hook_outputs(
        self,
        *,
        examples: List[dict],
        pooled_hidden: torch.Tensor,
        action_loss: torch.Tensor,
    ) -> tuple[dict, dict]:
        hook_cfg = _cfg_get(getattr(self.config, "trainer", None), "optional_loss_hooks", None)
        if not _cfg_enabled(hook_cfg, "enabled", default=False):
            return {}, {}

        pooled_hidden = pooled_hidden.to(dtype=action_loss.dtype)
        output_dict: dict[str, torch.Tensor] = {}
        debug_metrics: dict[str, float] = {}
        a_predictions = self.a_module_interface.predict(
            pooled_hidden=pooled_hidden,
            examples=examples,
            action_loss=action_loss,
            chunk_len=self.chunk_len,
        )
        debug_metrics.update(getattr(a_predictions, "debug_metrics", {}))
        # Standalone in-loop mode may attach runtime `a_outputs` during predict().
        # Build targets afterwards so embed supervision can consume that payload.
        targets = build_optional_hook_targets(
            examples,
            chunk_len=self.chunk_len,
            embedding_dim=self.a_embed_dim,
            device=action_loss.device,
            dtype=action_loss.dtype,
        )

        a_cfg = _cfg_get(hook_cfg, "a_loss", None)
        if _cfg_enabled(a_cfg, "enabled", default=False):
            a_key = str(_cfg_get(a_cfg, "key", "a_loss"))
            risk_weight = float(_cfg_get(a_cfg, "risk_weight", 1.0))
            trigger_weight = float(_cfg_get(a_cfg, "trigger_weight", 1.0))
            embed_weight = float(_cfg_get(a_cfg, "embed_weight", 1.0))
            embed_enabled = _cfg_enabled(a_cfg, "embed_enabled", default=True)

            risk_pred = a_predictions.risk_pred
            trigger_logit = a_predictions.trigger_logit
            embed_pred = self.a_embed_head(pooled_hidden)
            risk_loss_component = action_loss.new_zeros(())
            trigger_loss_component = action_loss.new_zeros(())
            embed_loss_component = action_loss.new_zeros(())
            risk_target_count = 0
            trigger_target_count = 0
            embed_target_count = 0

            if targets["risk_mask"].any():
                risk_mask = targets["risk_mask"]
                risk_loss_component = F.mse_loss(
                    risk_pred[risk_mask],
                    targets["risk_score"][risk_mask],
                )
                risk_target_count += int(risk_mask.sum().item())
            if targets["trigger_mask"].any():
                trigger_mask = targets["trigger_mask"]
                trigger_loss_component = F.binary_cross_entropy_with_logits(
                    trigger_logit[trigger_mask],
                    targets["trigger_label"][trigger_mask].clamp(0.0, 1.0),
                )
                trigger_target_count += int(trigger_mask.sum().item())
            if embed_enabled and targets["embedding_mask"].any():
                embed_mask = targets["embedding_mask"]
                embed_loss_component = F.mse_loss(
                    embed_pred[embed_mask],
                    targets["dynamic_embedding"][embed_mask],
                )
                embed_target_count += int(embed_mask.sum().item())

            weighted_risk_loss = risk_weight * risk_loss_component
            weighted_trigger_loss = trigger_weight * trigger_loss_component
            weighted_embed_loss = embed_weight * embed_loss_component if embed_enabled else action_loss.new_zeros(())
            a_loss = weighted_risk_loss + weighted_trigger_loss + weighted_embed_loss

            # P1 (v0.9.2): trigger-region consistency regularization
            consist_weight = float(_cfg_get(a_cfg, "consist_weight", 0.1))
            consist_loss = action_loss.new_zeros(())
            region_logits_for_consist = a_predictions.region_logits
            if consist_weight > 0 and region_logits_for_consist is not None:
                trigger_prob = torch.sigmoid(trigger_logit)
                region_prob_mean = torch.sigmoid(region_logits_for_consist).mean(dim=-1)
                margin = region_prob_mean - trigger_prob - 0.1
                consist_loss = consist_weight * (F.relu(margin) ** 2).mean()
                a_loss = a_loss + consist_loss
            output_dict["a_loss_consist"] = consist_loss

            output_dict["a_loss_risk"] = weighted_risk_loss
            output_dict["a_loss_trigger"] = weighted_trigger_loss
            output_dict["a_loss_embed"] = weighted_embed_loss
            output_dict[a_key] = a_loss
            debug_metrics["debug/a_loss_risk_target_count"] = float(risk_target_count)
            debug_metrics["debug/a_loss_trigger_target_count"] = float(trigger_target_count)
            debug_metrics["debug/a_loss_embed_target_count"] = float(embed_target_count)
            debug_metrics["debug/a_loss_embed_enabled"] = 1.0 if embed_enabled else 0.0
            debug_metrics["debug/a_loss_embed_target_coverage"] = float(
                embed_target_count / max(1, int(targets["embedding_mask"].numel()))
            )
            debug_metrics["debug/a_loss_target_count"] = float(
                risk_target_count + trigger_target_count + embed_target_count
            )

        corrective_cfg = _cfg_get(hook_cfg, "corrective_loss", None)
        if _cfg_enabled(corrective_cfg, "enabled", default=False):
            corrective_key = str(_cfg_get(corrective_cfg, "key", "corrective_loss"))
            delta_norm_weight = float(_cfg_get(corrective_cfg, "delta_norm_weight", 1.0))
            correction_mask_weight = float(_cfg_get(corrective_cfg, "correction_mask_weight", 1.0))
            region_prior_weight = float(_cfg_get(corrective_cfg, "region_prior_weight", 0.5))

            delta_pred = a_predictions.delta_pred
            region_logits = a_predictions.region_logits
            delta_component = action_loss.new_zeros(())
            region_component = action_loss.new_zeros(())
            delta_target_count = 0
            correction_mask_target_count = 0
            region_prior_target_count = 0
            region_trigger_filtered_count = 0

            if targets["delta_mask"].any():
                delta_mask = targets["delta_mask"]
                delta_component = delta_norm_weight * F.mse_loss(
                    delta_pred[delta_mask],
                    targets["delta_action_norm"][delta_mask],
                )
                delta_target_count += int(delta_mask.sum().item())

            # P0 (v0.9.2): region loss conditional on trigger_label=1
            trigger_positive = targets["trigger_label"] > 0.5

            if targets["correction_mask_mask"].any():
                correction_mask = targets["correction_mask_mask"] & trigger_positive
                if correction_mask.any():
                    region_component = region_component + correction_mask_weight * F.binary_cross_entropy_with_logits(
                        region_logits[correction_mask],
                        targets["correction_mask"][correction_mask].clamp(0.0, 1.0),
                    )
                    correction_mask_target_count += int(correction_mask.sum().item())
                region_trigger_filtered_count += int(
                    (targets["correction_mask_mask"] & ~trigger_positive).sum().item()
                )

            if targets["region_prior_mask"].any():
                region_mask = targets["region_prior_mask"] & trigger_positive
                if region_mask.any():
                    region_prob = torch.sigmoid(region_logits[region_mask])
                    region_component = region_component + region_prior_weight * F.mse_loss(
                        region_prob,
                        targets["region_prior"][region_mask].clamp(0.0, 1.0),
                    )
                    region_prior_target_count += int(region_mask.sum().item())
                region_trigger_filtered_count += int(
                    (targets["region_prior_mask"] & ~trigger_positive).sum().item()
                )

            corrective_loss = delta_component + region_component
            output_dict["corrective_loss_delta"] = delta_component
            output_dict["corrective_loss_region"] = region_component
            output_dict[corrective_key] = corrective_loss
            debug_metrics["debug/corrective_loss_delta_target_count"] = float(delta_target_count)
            debug_metrics["debug/corrective_loss_region_mask_target_count"] = float(
                correction_mask_target_count
            )
            debug_metrics["debug/corrective_loss_region_prior_target_count"] = float(
                region_prior_target_count
            )
            debug_metrics["debug/corrective_loss_region_trigger_filtered_count"] = float(
                region_trigger_filtered_count
            )
            debug_metrics["debug/corrective_loss_target_count"] = float(
                delta_target_count + correction_mask_target_count + region_prior_target_count
            )

        return output_dict, debug_metrics
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        runtime_mp = getattr(self, "_starvla_runtime_mixed_precision", None)
        if runtime_mp is not None:
            setattr(self.qwen_vl_interface, "_starvla_runtime_mixed_precision", runtime_mp)
        with _runtime_autocast_context(self):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # 取与 DiT 层数匹配的最后 N 层隐藏态，按层喂给 DiT
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", enabled=False):
            action_model_dtype = ACTION_PATH_DTYPE
            try:
                action_model_dtype = next(self.action_model.parameters()).dtype
            except Exception:
                pass
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=action_model_dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            repeated_diffusion_steps = 2 # NO repeat for big action FM
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # 对每层特征做 repeat
            vl_embs_list_repeated = [
                h.to(dtype=action_model_dtype).repeat(repeated_diffusion_steps, 1, 1)
                for h in vl_embs_list
            ]
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=action_model_dtype
                )
                expected_sdim = getattr(self.config.framework.action_model, "state_dim", None)
                if expected_sdim and state.shape[-1] > expected_sdim:
                    state = state[..., :expected_sdim]
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)


        output_dict = {"action_loss": action_loss}
        hook_output_dict, hook_debug_metrics = self._compute_optional_hook_outputs(
            examples=examples,
            pooled_hidden=base_hidden.to(dtype=action_model_dtype).mean(dim=1),
            action_loss=action_loss,
        )
        if hook_output_dict:
            output_dict.update(hook_output_dict)
        if hook_debug_metrics:
            output_dict["debug_metrics"] = hook_debug_metrics

        if self.corrective_flow_enabled:
            cf_trainer_cfg = _cfg_get(
                _cfg_get(
                    getattr(self.config, "trainer", None),
                    "optional_loss_hooks",
                    None,
                ),
                "corrective_flow",
                None,
            )
            if _cfg_enabled(cf_trainer_cfg, "enabled", default=False):
                noise_scale = float(_cfg_get(cf_trainer_cfg, "noise_scale", 0.05))

                if actions.shape[1] >= 2 * self.chunk_len:
                    a_prev = actions[:, : self.chunk_len, :]
                else:
                    a_prev = torch.zeros_like(actions_target)
                a_prev_noisy = a_prev + noise_scale * torch.randn_like(a_prev)

                pooled_for_cf = base_hidden.to(dtype=action_model_dtype).mean(dim=1)
                a_predictions_cf = self.a_module_interface.predict(
                    pooled_hidden=pooled_for_cf,
                    examples=examples,
                    action_loss=action_loss,
                    chunk_len=self.chunk_len,
                )
                trigger_prob = torch.sigmoid(a_predictions_cf.trigger_logit).detach().unsqueeze(-1)
                region_logits = a_predictions_cf.region_logits.detach()
                vl_embs_for_cf = base_hidden.to(dtype=action_model_dtype).detach()

                cf_loss, _, cf_debug = self.corrective_flow_head(
                    vl_embs=vl_embs_for_cf,
                    a_prev=a_prev_noisy.to(dtype=action_model_dtype),
                    a_gt=actions_target.to(dtype=action_model_dtype),
                    trigger_prob=trigger_prob.to(dtype=action_model_dtype),
                    region_logits=region_logits.to(dtype=action_model_dtype),
                )
                output_dict["corrective_flow_loss"] = cf_loss
                for k, v in cf_debug.items():
                    output_dict[k] = v

        return output_dict

    @torch.inference_mode()
    def predict_action( # TODO align  predict_action with forward, make api more flexible
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        from deployment.model_server.tools.image_tools import to_pil_preserve
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        runtime_mp = getattr(self, "_starvla_runtime_mixed_precision", None)
        if runtime_mp is not None:
            setattr(self.qwen_vl_interface, "_starvla_runtime_mixed_precision", runtime_mp)
        with _runtime_autocast_context(self):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        action_model_dtype = ACTION_PATH_DTYPE
        try:
            action_model_dtype = next(self.action_model.parameters()).dtype
        except Exception:
            pass
        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=action_model_dtype)
            if state is not None
            else None
        )
        if state is not None:
            expected_sdim = getattr(self.config.framework.action_model, "state_dim", None)
            if expected_sdim and state.shape[-1] > expected_sdim:
                state = state[..., :expected_sdim]
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", enabled=False):
            action_vl_embs = [h.to(dtype=action_model_dtype) for h in vl_embs_list]
            pred_actions = self.action_model.predict_action(action_vl_embs, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().to(dtype=torch.float32).cpu().numpy()

        if self.corrective_flow_enabled:
            with torch.autocast("cuda", enabled=False):
                ami = self.a_module_interface
                ami_is_module = isinstance(ami, nn.Module)
                orig_ami_dtype = None
                orig_cf_dtype = next(self.corrective_flow_head.parameters()).dtype
                try:
                    if ami_is_module:
                        orig_ami_dtype = next(ami.parameters()).dtype
                        ami.float()
                    self.corrective_flow_head.float()

                    pooled_cf = base_hidden.float().mean(dim=1)
                    a_predictions_cf = ami.predict(
                        pooled_hidden=pooled_cf,
                        examples=examples,
                        action_loss=torch.zeros(1, device=base_hidden.device),
                        chunk_len=self.chunk_len,
                    )
                    trigger_prob = torch.sigmoid(a_predictions_cf.trigger_logit)
                    if trigger_prob.max().item() > self.corrective_flow_trigger_threshold:
                        region_logits = a_predictions_cf.region_logits
                        a_prev_for_cf = (
                            self._cf_prev_chunk.to(device=pred_actions.device)
                            if self._cf_prev_chunk is not None
                            else torch.zeros_like(pred_actions)
                        )
                        a_corrected = self.corrective_flow_head.predict(
                            vl_embs=base_hidden.float(),
                            a_base=a_prev_for_cf.float(),
                            trigger_prob=trigger_prob.unsqueeze(-1).float(),
                            region_logits=region_logits.float(),
                        )
                        normalized_actions = a_corrected.detach().to(dtype=torch.float32).cpu().numpy()
                    self._cf_prev_chunk = pred_actions.detach().clone()
                except RuntimeError as e:
                    logger.warning("Corrective flow inference skipped (dtype): %s", e)
                finally:
                    if ami_is_module and orig_ami_dtype is not None:
                        ami.to(dtype=orig_ami_dtype)
                    self.corrective_flow_head.to(dtype=orig_cf_dtype)

        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    

    model = Qwen_PI(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)


    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
