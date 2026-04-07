# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

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
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

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
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        llm_hidden_size = self.qwen_vl_interface.model.config.hidden_size
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
        self.a_module_interface = build_a_module_interface(
            config=self.config,
            chunk_len=self.chunk_len,
            a_risk_head=self.a_risk_head,
            a_trigger_head=self.a_trigger_head,
            corrective_delta_head=self.corrective_delta_head,
            corrective_region_head=self.corrective_region_head,
        )

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
        targets = build_optional_hook_targets(
            examples,
            chunk_len=self.chunk_len,
            device=action_loss.device,
            dtype=action_loss.dtype,
        )
        output_dict: dict[str, torch.Tensor] = {}
        debug_metrics: dict[str, float] = {}
        a_predictions = self.a_module_interface.predict(
            pooled_hidden=pooled_hidden,
            examples=examples,
            action_loss=action_loss,
            chunk_len=self.chunk_len,
        )
        debug_metrics.update(getattr(a_predictions, "debug_metrics", {}))

        a_cfg = _cfg_get(hook_cfg, "a_loss", None)
        if _cfg_enabled(a_cfg, "enabled", default=False):
            a_key = str(_cfg_get(a_cfg, "key", "a_loss"))
            risk_weight = float(_cfg_get(a_cfg, "risk_weight", 1.0))
            trigger_weight = float(_cfg_get(a_cfg, "trigger_weight", 1.0))

            risk_pred = a_predictions.risk_pred
            trigger_logit = a_predictions.trigger_logit
            a_loss = action_loss.new_zeros(())
            target_count = 0

            if targets["risk_mask"].any():
                risk_mask = targets["risk_mask"]
                a_loss = a_loss + risk_weight * F.mse_loss(
                    risk_pred[risk_mask],
                    targets["risk_score"][risk_mask],
                )
                target_count += int(risk_mask.sum().item())
            if targets["trigger_mask"].any():
                trigger_mask = targets["trigger_mask"]
                a_loss = a_loss + trigger_weight * F.binary_cross_entropy_with_logits(
                    trigger_logit[trigger_mask],
                    targets["trigger_label"][trigger_mask].clamp(0.0, 1.0),
                )
                target_count += int(trigger_mask.sum().item())

            output_dict[a_key] = a_loss
            debug_metrics["debug/a_loss_target_count"] = float(target_count)

        corrective_cfg = _cfg_get(hook_cfg, "corrective_loss", None)
        if _cfg_enabled(corrective_cfg, "enabled", default=False):
            corrective_key = str(_cfg_get(corrective_cfg, "key", "corrective_loss"))
            delta_norm_weight = float(_cfg_get(corrective_cfg, "delta_norm_weight", 1.0))
            correction_mask_weight = float(_cfg_get(corrective_cfg, "correction_mask_weight", 1.0))
            region_prior_weight = float(_cfg_get(corrective_cfg, "region_prior_weight", 0.5))

            delta_pred = a_predictions.delta_pred
            region_logits = a_predictions.region_logits
            corrective_loss = action_loss.new_zeros(())
            target_count = 0

            if targets["delta_mask"].any():
                delta_mask = targets["delta_mask"]
                corrective_loss = corrective_loss + delta_norm_weight * F.mse_loss(
                    delta_pred[delta_mask],
                    targets["delta_action_norm"][delta_mask],
                )
                target_count += int(delta_mask.sum().item())

            if targets["correction_mask_mask"].any():
                correction_mask = targets["correction_mask_mask"]
                corrective_loss = corrective_loss + correction_mask_weight * F.binary_cross_entropy_with_logits(
                    region_logits[correction_mask],
                    targets["correction_mask"][correction_mask].clamp(0.0, 1.0),
                )
                target_count += int(correction_mask.sum().item())

            if targets["region_prior_mask"].any():
                region_mask = targets["region_prior_mask"]
                region_prob = torch.sigmoid(region_logits[region_mask])
                corrective_loss = corrective_loss + region_prior_weight * F.mse_loss(
                    region_prob,
                    targets["region_prior"][region_mask].clamp(0.0, 1.0),
                )
                target_count += int(region_mask.sum().item())

            output_dict[corrective_key] = corrective_loss
            debug_metrics["debug/corrective_loss_target_count"] = float(target_count)

        return output_dict, debug_metrics
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)


        output_dict = {"action_loss": action_loss}
        hook_output_dict, hook_debug_metrics = self._compute_optional_hook_outputs(
            examples=examples,
            pooled_hidden=last_hidden.mean(dim=1),
            action_loss=action_loss,
        )
        if hook_output_dict:
            output_dict.update(hook_output_dict)
        if hook_debug_metrics:
            output_dict["debug_metrics"] = hook_debug_metrics

        return output_dict

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # forward model with dataloader
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        # break

    action = model.predict_action(examples=batch)
    print("Finished")
