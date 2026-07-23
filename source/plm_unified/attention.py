"""Adapter, cross-attention, and pooling blocks for the unified model."""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.network(hidden)
        return output.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        model_dim: int = 128,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.feedforward_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        peptide: torch.Tensor,
        receptor: torch.Tensor,
        peptide_mask: torch.Tensor,
        receptor_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        peptide_mask = peptide_mask.bool()
        receptor_mask = receptor_mask.bool()
        if (~peptide_mask.any(dim=1)).any():
            raise ValueError("Every peptide sequence must contain at least one residue")
        if (~receptor_mask.any(dim=1)).any():
            raise ValueError("Every receptor sequence must contain at least one residue")

        attended, weights = self.attention(
            query=peptide,
            key=receptor,
            value=receptor,
            key_padding_mask=~receptor_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden = self.attention_norm(
            peptide + self.attention_dropout(attended)
        )
        hidden = self.feedforward_norm(hidden + self.feedforward(hidden))
        hidden = hidden.masked_fill(~peptide_mask.unsqueeze(-1), 0.0)
        return hidden, weights if return_attention else None


class MaskedAttentionPooling(nn.Module):
    def __init__(self, model_dim: int = 128):
        super().__init__()
        self.score = nn.Linear(model_dim, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mask = mask.bool()
        if (~mask.any(dim=1)).any():
            raise ValueError("Cannot pool a sequence with no real residues")
        scores = self.score(sequence).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights if return_weights else None


class BindingClassifier(nn.Module):
    def __init__(self, model_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.network(pooled)
