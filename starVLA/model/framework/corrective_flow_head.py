"""
CorrectiveFlowHead — one-step corrective flow for StarVLA (RTC paradigm v1.1).

Takes the base-policy action chunk as a near-correct starting point and
predicts a velocity field via a small DiT to refine it in a single ODE step.

RTC (Region-weighted Training-time Conditioning):
  - Training: region_logits from the A-module fusion head weight the loss gradient,
    but are NOT input features to the CF model.
  - Inference: CF only needs a_prev — no trigger gating, no region gating,
    no A-module / VDPM dependency.
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
    One-step corrective flow head (RTC paradigm v1.1).

    Architecture
    ------------
    ActionEncoder → encode (a_prev, timestep) into hidden features
    DiT (small)   → self-attend over action tokens (cross-attn optional via config)
    ActionDecoder → project back to action_dim velocity

    RTC: MetadataEncoder is retained for weight-compat but NOT called in forward path.
    region_logits only affect the loss gradient, never the model input.
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
        # Retained for checkpoint backward-compat; not called in RTC forward path.
        self.metadata_encoder = MetadataEncoder(
            input_dim=1 + chunk_len, hidden_dim=hidden_dim
        )

        # BD-7: vl_hidden_dim=0 → cross_attention_dim=None (self-attn only, recommended)
        effective_cross_attn_dim = vl_hidden_dim if vl_hidden_dim > 0 else None

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
            cross_attention_dim=effective_cross_attn_dim,
        )

        self.action_decoder = MLP(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=action_dim
        )

        self._t_fixed = 0.9
        self._t_discretized = int(self._t_fixed * 1000)

        logger.info(
            "CorrectiveFlowHead(RTC): action_dim=%d chunk_len=%d hidden=%d "
            "vl_hidden=%d(cross_attn=%s) layers=%d heads=%d region_loss_w=%.2f params=%d",
            action_dim,
            chunk_len,
            hidden_dim,
            vl_hidden_dim,
            effective_cross_attn_dim is not None,
            num_layers,
            num_heads,
            region_loss_weight,
            sum(p.numel() for p in self.parameters()),
        )

    def _encode_and_attend(
        self,
        a_input: torch.Tensor,
        vl_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared encode → DiT → decode pipeline (RTC: no metadata token)."""
        B = a_input.shape[0]
        device = a_input.device

        timesteps = torch.full(
            (B,), self._t_discretized, device=device, dtype=torch.long
        )

        action_features = self.action_encoder(a_input, timesteps)  # [B, C, hidden]

        # RTC: no MetadataEncoder — all tokens are action tokens
        sa_embs = action_features  # [B, C, hidden]

        temb = self.dit.timestep_encoder(timesteps)

        hidden_states = sa_embs
        for block in self.dit.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=vl_embs,  # None when BD-7=self-attn only
                temb=temb,
            )

        shift, scale = self.dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = (
            self.dit.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        hidden_states = self.dit.proj_out_2(hidden_states)

        pred_velocity = self.action_decoder(hidden_states)  # [B, C, action_dim]
        return pred_velocity

    def forward(
        self,
        a_prev: torch.Tensor,
        a_gt: torch.Tensor,
        region_logits: torch.Tensor,
        vl_embs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Training forward (RTC paradigm).

        Args:
            a_prev:        [B, C, action_dim]  previous chunk (GT + noise)
            a_gt:          [B, C, action_dim]  current chunk GT
            region_logits: [B, C]             from fusion head (detached), loss weighting only
            vl_embs:       [B, S, vl_hidden_dim] optional VLM features (None if BD-7=no cross-attn)

        Returns:
            (loss, pred_velocity, debug_dict)
        """
        velocity_gt = a_gt - a_prev

        pred_velocity = self._encode_and_attend(a_prev, vl_embs)

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
        a_prev: torch.Tensor,
        vl_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inference: one-step corrective flow (RTC paradigm).

        No trigger gating, no region gating. The model has learned via
        region-weighted loss to self-determine correction magnitude.

        Args:
            a_prev:  [B, C, action_dim]  previous base-policy prediction
            vl_embs: [B, S, vl_hidden_dim] optional VLM features

        Returns:
            a_corrected: [B, C, action_dim]
        """
        pred_velocity = self._encode_and_attend(a_prev, vl_embs)
        a_corrected = a_prev + pred_velocity
        return a_corrected
