"""
CorrectiveFlowHead — one-step corrective flow for StarVLA.

Takes the base-policy action chunk as a near-correct starting point and
predicts a velocity field via a small DiT to refine it in a single ODE step.
Region logits from the A-module gate which steps receive correction.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    ActionEncoder,
    MLP,
)

logger = logging.getLogger(__name__)


class MetadataEncoder(nn.Module):
    """Encode trigger_prob + region_logits into a single conditioning token."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CorrectiveFlowHead(nn.Module):
    """
    One-step corrective flow head.

    Architecture
    ------------
    ActionEncoder  → encode (a_prev, timestep) into hidden features
    MetadataEncoder→ encode (trigger_prob ‖ region_logits) into a conditioning token
    DiT (small)    → cross-attend to VLM features, self-attend over action+meta tokens
    ActionDecoder  → project back to action_dim velocity
    """

    def __init__(
        self,
        action_dim: int,
        chunk_len: int,
        hidden_dim: int,
        vl_hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 16,
        region_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.hidden_dim = hidden_dim
        self.vl_hidden_dim = vl_hidden_dim
        self.region_loss_weight = region_loss_weight

        self.action_encoder = ActionEncoder(
            action_dim=action_dim, hidden_size=hidden_dim
        )
        self.metadata_encoder = MetadataEncoder(
            input_dim=1 + chunk_len, hidden_dim=hidden_dim
        )

        attention_head_dim = 64
        self.dit = DiT(
            num_attention_heads=num_heads,
            attention_head_dim=attention_head_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            upcast_attention=False,
            norm_type="ada_norm",
            norm_elementwise_affine=False,
            norm_eps=1e-5,
            max_num_positional_embeddings=512,
            compute_dtype=torch.float32,
            final_dropout=True,
            positional_embeddings=None,
            interleave_self_attention=False,
            cross_attention_dim=vl_hidden_dim,
        )

        self.action_decoder = MLP(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=action_dim
        )

        self._t_fixed = 0.9
        self._t_discretized = int(self._t_fixed * 1000)

        logger.info(
            "CorrectiveFlowHead: action_dim=%d chunk_len=%d hidden=%d "
            "vl_hidden=%d layers=%d heads=%d region_loss_w=%.2f params=%d",
            action_dim,
            chunk_len,
            hidden_dim,
            vl_hidden_dim,
            num_layers,
            num_heads,
            region_loss_weight,
            sum(p.numel() for p in self.parameters()),
        )

    def _encode_and_attend(
        self,
        vl_embs: torch.Tensor,
        a_input: torch.Tensor,
        trigger_prob: torch.Tensor,
        region_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Shared encode → DiT → decode pipeline for forward / predict."""
        B = a_input.shape[0]
        device = a_input.device

        timesteps = torch.full(
            (B,), self._t_discretized, device=device, dtype=torch.long
        )

        action_features = self.action_encoder(a_input, timesteps)  # [B, C, hidden]

        meta_input = torch.cat([trigger_prob, region_logits], dim=-1)  # [B, 1+C]
        meta_features = self.metadata_encoder(meta_input).unsqueeze(1)  # [B, 1, hidden]

        sa_embs = torch.cat([meta_features, action_features], dim=1)  # [B, 1+C, hidden]

        temb = self.dit.timestep_encoder(timesteps)

        hidden_states = sa_embs
        for block in self.dit.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=vl_embs,
                temb=temb,
            )

        shift, scale = self.dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = (
            self.dit.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        hidden_states = self.dit.proj_out_2(hidden_states)

        action_hidden = hidden_states[:, 1:, :]  # drop metadata token
        pred_velocity = self.action_decoder(action_hidden)  # [B, C, action_dim]
        return pred_velocity

    def forward(
        self,
        vl_embs: torch.Tensor,
        a_prev: torch.Tensor,
        a_gt: torch.Tensor,
        trigger_prob: torch.Tensor,
        region_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Training forward.

        Args:
            vl_embs:       [B, S, vl_hidden_dim] VLM features (detached)
            a_prev:        [B, C, action_dim]     previous chunk (GT + noise)
            a_gt:          [B, C, action_dim]     current chunk GT
            trigger_prob:  [B, 1]                 sigmoid(trigger_logit) (detached)
            region_logits: [B, C]                 region head output (detached)

        Returns:
            (loss, pred_velocity, debug_dict)
        """
        velocity_gt = a_gt - a_prev

        pred_velocity = self._encode_and_attend(
            vl_embs, a_prev, trigger_prob, region_logits
        )

        region_weight = torch.sigmoid(region_logits).unsqueeze(-1)  # [B, C, 1]
        per_step_loss = (pred_velocity - velocity_gt) ** 2
        base_loss = per_step_loss.mean()
        region_loss = (region_weight * per_step_loss).mean()
        loss = base_loss + self.region_loss_weight * region_loss

        debug = {
            "corrective_flow_base_loss": base_loss.detach(),
            "corrective_flow_region_loss": region_loss.detach(),
        }
        return loss, pred_velocity, debug

    @torch.no_grad()
    def predict(
        self,
        vl_embs: torch.Tensor,
        a_base: torch.Tensor,
        trigger_prob: torch.Tensor,
        region_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference: one-step corrective flow.

        Args:
            vl_embs:       [B, S, vl_hidden_dim]
            a_base:        [B, C, action_dim]  base-policy output
            trigger_prob:  [B, 1]
            region_logits: [B, C]

        Returns:
            a_corrected:   [B, C, action_dim]
        """
        pred_velocity = self._encode_and_attend(
            vl_embs, a_base, trigger_prob, region_logits
        )
        region_gate = torch.clamp(
            torch.sigmoid(region_logits) - 0.3, min=0.0
        ).unsqueeze(-1)  # [B, C, 1]; zero below sigmoid=0.3
        a_corrected = a_base + region_gate * pred_velocity
        return a_corrected
