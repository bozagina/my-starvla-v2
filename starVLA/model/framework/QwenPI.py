# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""
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

        a_cfg = _cfg_get(hook_cfg, "a_loss", None)
        if _cfg_enabled(a_cfg, "enabled", default=False):
            a_key = str(_cfg_get(a_cfg, "key", "a_loss"))
            risk_weight = float(_cfg_get(a_cfg, "risk_weight", 1.0))
            trigger_weight = float(_cfg_get(a_cfg, "trigger_weight", 1.0))

            risk_pred = self.a_risk_head(pooled_hidden).squeeze(-1)
            trigger_logit = self.a_trigger_head(pooled_hidden).squeeze(-1)
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

            delta_pred = self.corrective_delta_head(pooled_hidden).squeeze(-1)
            region_logits = self.corrective_region_head(pooled_hidden)
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            repeated_diffusion_steps = 2 # NO repeat for big action FM
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # 对每层特征做 repeat
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)


        output_dict = {"action_loss": action_loss}
        hook_output_dict, hook_debug_metrics = self._compute_optional_hook_outputs(
            examples=examples,
            pooled_hidden=base_hidden.mean(dim=1),
            action_loss=action_loss,
        )
        if hook_output_dict:
            output_dict.update(hook_output_dict)
        if hook_debug_metrics:
            output_dict["debug_metrics"] = hook_debug_metrics

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
        with torch.autocast("cuda", dtype=torch.bfloat16):
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

        state = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
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
