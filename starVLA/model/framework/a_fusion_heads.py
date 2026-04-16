from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AModuleRuntimeBatch:
    dynamic_embedding: torch.Tensor
    embedding_mask: torch.Tensor
    rows_with_payload: int
    rows_with_embedding: int
    rows_missing_embedding: int
    fallback_to_precomputed_count: int
    inloop_runtime_model_enabled: bool


class Region15ToChunkAdapter(nn.Module):
    """Deterministic adapter from contract region logits [B, 15] to train-time chunk logits [B, chunk_len]."""

    def __init__(self, *, chunk_len: int) -> None:
        super().__init__()
        self.chunk_len = int(chunk_len)
        if self.chunk_len <= 0:
            raise ValueError(f"chunk_len must be > 0, got {chunk_len}")

    def forward(self, region_logits_contract15: torch.Tensor) -> torch.Tensor:
        if region_logits_contract15.ndim != 2 or region_logits_contract15.shape[1] != 15:
            raise ValueError(
                "region_logits_contract15 must have shape [B, 15], "
                f"got {tuple(region_logits_contract15.shape)}"
            )
        if self.chunk_len == 15:
            return region_logits_contract15
        x = region_logits_contract15.unsqueeze(1)  # [B, 1, 15]
        y = F.interpolate(x, size=self.chunk_len, mode="linear", align_corners=False)
        return y.squeeze(1)


class _CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, *, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q1 = self.norm1(queries)
        self_attn_out, _ = self.self_attn(q1, q1, q1, need_weights=False)
        queries = queries + self.dropout(self_attn_out)

        q2 = self.norm2(queries)
        cross_attn_out, _ = self.cross_attn(q2, memory, memory, need_weights=False)
        queries = queries + self.dropout(cross_attn_out)

        q3 = self.norm3(queries)
        queries = queries + self.dropout(self.ffn(q3))
        return queries


class CrossAttentionFusionAHead(nn.Module):
    """Q-Bridge cross-attention fusion head for A-module outputs."""

    def __init__(
        self,
        *,
        pooled_hidden_dim: int,
        chunk_len: int,
        vdpm_embedding_dim: int = 16,
        model_dim: int = 256,
        num_heads: int = 8,
        decoder_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pooled_hidden_dim = int(pooled_hidden_dim)
        self.vdpm_embedding_dim = int(vdpm_embedding_dim)
        self.model_dim = int(model_dim)
        self.decoder_layers = int(decoder_layers)
        self.slot_count = 4
        if self.vdpm_embedding_dim % self.slot_count != 0:
            raise ValueError(
                f"vdpm_embedding_dim must be divisible by {self.slot_count}, got {self.vdpm_embedding_dim}"
            )
        self.slot_width = self.vdpm_embedding_dim // self.slot_count
        self.dynamic_query_count = 4

        self.q_proj = nn.Linear(self.pooled_hidden_dim, self.model_dim)
        self.vdpm_global_proj = nn.Linear(self.vdpm_embedding_dim, self.model_dim)
        self.slot_proj = nn.Linear(self.slot_width, self.model_dim)

        self.global_query = nn.Parameter(torch.zeros(1, self.model_dim))
        self.region_queries = nn.Parameter(torch.zeros(15, self.model_dim))
        self.dynamic_queries = nn.Parameter(torch.zeros(self.dynamic_query_count, self.model_dim))
        nn.init.normal_(self.global_query, std=0.02)
        nn.init.normal_(self.region_queries, std=0.02)
        nn.init.normal_(self.dynamic_queries, std=0.02)

        self.decoder = nn.ModuleList(
            [
                _CrossAttentionDecoderLayer(
                    model_dim=self.model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(max(1, self.decoder_layers))
            ]
        )

        self.risk_head = nn.Linear(self.model_dim, 1)
        self.trigger_head = nn.Linear(self.model_dim, 1)
        self.delta_head = nn.Linear(self.model_dim, 1)
        self.region_head = nn.Linear(self.model_dim, 1)
        self.dynamic_head = nn.Linear(self.model_dim, self.slot_width)
        self.region_adapter = Region15ToChunkAdapter(chunk_len=chunk_len)

    def _normalize_runtime_embedding(
        self,
        embedding: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError(f"runtime dynamic_embedding must be rank-2, got {tuple(embedding.shape)}")
        emb = embedding.to(device=device, dtype=dtype)
        if emb.shape[1] == self.vdpm_embedding_dim:
            return emb
        if emb.shape[1] > self.vdpm_embedding_dim:
            return emb[:, : self.vdpm_embedding_dim]
        pad = emb.new_zeros((emb.shape[0], self.vdpm_embedding_dim - emb.shape[1]))
        return torch.cat([emb, pad], dim=1)

    def forward(
        self,
        *,
        pooled_hidden: torch.Tensor,
        runtime_batch: AModuleRuntimeBatch,
    ) -> dict[str, torch.Tensor]:
        if pooled_hidden.ndim != 2:
            raise ValueError(f"pooled_hidden must be rank-2 [B, H], got {tuple(pooled_hidden.shape)}")
        batch_size = int(pooled_hidden.shape[0])
        if runtime_batch.dynamic_embedding.shape[0] != batch_size:
            raise ValueError(
                "runtime embedding batch mismatch: "
                f"pooled_hidden batch={batch_size}, runtime batch={runtime_batch.dynamic_embedding.shape[0]}"
            )

        device = pooled_hidden.device
        dtype = pooled_hidden.dtype
        vdpm_embed = self._normalize_runtime_embedding(
            runtime_batch.dynamic_embedding,
            device=device,
            dtype=dtype,
        )

        q_token = self.q_proj(pooled_hidden).unsqueeze(1)  # [B, 1, D]
        vdpm_global_token = self.vdpm_global_proj(vdpm_embed).unsqueeze(1)  # [B, 1, D]
        vdpm_slots = self.slot_proj(vdpm_embed.view(batch_size, self.slot_count, self.slot_width))  # [B, 4, D]
        memory = torch.cat([q_token, vdpm_global_token, vdpm_slots], dim=1)  # [B, 6, D]

        learned_queries = torch.cat(
            [
                self.global_query,
                self.region_queries,
                self.dynamic_queries,
            ],
            dim=0,
        )
        queries = learned_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)

        for layer in self.decoder:
            queries = layer(queries, memory)

        global_query = queries[:, 0]  # [B, D]
        region_queries = queries[:, 1 : 1 + 15]  # [B, 15, D]
        dynamic_queries = queries[:, 1 + 15 :]  # [B, 4, D]

        risk_pred = self.risk_head(global_query).squeeze(-1)
        trigger_logit = self.trigger_head(global_query).squeeze(-1)
        delta_pred = F.softplus(self.delta_head(global_query).squeeze(-1))

        region_logits_contract15 = self.region_head(region_queries).squeeze(-1)  # [B, 15]
        region_logits_train = self.region_adapter(region_logits_contract15)  # [B, chunk_len]

        dynamic_embedding_pred = self.dynamic_head(dynamic_queries).reshape(batch_size, -1)
        dynamic_embedding_pred = dynamic_embedding_pred[:, : self.vdpm_embedding_dim]

        return {
            "risk_pred": risk_pred,
            "trigger_logit": trigger_logit,
            "delta_pred": delta_pred,
            "region_logits_contract15": region_logits_contract15,
            "region_logits_train": region_logits_train,
            "dynamic_embedding_pred": dynamic_embedding_pred,
        }
