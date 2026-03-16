"""
PatchTST for multivariate time series classification.

Reference: Nie et al., ICLR 2023 — https://arxiv.org/abs/2211.14730

Architecture
------------
1. RevIN       — per-instance normalisation, removes residual instance-level flux scale
                 variation that persists after the dataset-level z-normalisation applied
                 in the data pipeline (individual LSST objects can differ greatly in
                 absolute brightness even within the same normalised passband)
2. Patch embed — channel-independent unfold -> linear projection + learnable positional enc.
3. Transformer — shared weights across channels (channel-independent, no cross-channel leakage)
4. GAP         — mean over patches per channel  ->  (B*C, d_model)
5. Head        — concat all channel representations  ->  2-layer MLP  ->  logits
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch import Tensor


class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al., 2022)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features)) if affine else None
        self.beta  = nn.Parameter(torch.zeros(num_features)) if affine else None

    def _update_stats(self, x: Tensor) -> None:
        self._mean = x.mean(dim=1, keepdim=True)          # (B, 1, C)
        self._std  = x.std(dim=1, keepdim=True) + self.eps

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._update_stats(x)
            x = (x - self._mean) / self._std
            if self.gamma is not None:
                x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.gamma is not None:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self._std + self._mean
        return x


class PatchTSTClassifier(nn.Module):
    """
    Channel-independent PatchTST encoder for multivariate time series classification.

    Input : x    (B, T, C)  — time series
            mask (B, T, C)  — binary observation mask (accepted for API consistency, not used)
    Output: logits (B, num_classes)
    """

    is_foundation_model = False

    def __init__(
        self,
        seq_len: int = 36,
        n_channels: int = 6,
        num_classes: int = 14,
        patch_len: int = 4,
        stride: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        dropout_head: float = 0.2,
    ) -> None:
        super().__init__()

        self.seq_len    = seq_len
        self.n_channels = n_channels
        self.patch_len  = patch_len
        self.stride     = stride
        self.d_model    = d_model

        num_patches = (seq_len - patch_len) // stride + 1
        self.num_patches = num_patches

        self.revin = RevIN(n_channels)

        # Patch projection and positional encoding (shared across channels)
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_enc    = nn.Parameter(torch.empty(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        self.drop_emb   = nn.Dropout(dropout)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model * n_channels, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(d_ff, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _patch_and_embed(self, x: Tensor) -> Tensor:
        """(B, T, C) -> (B*C, num_patches, d_model)."""
        B, T, C = x.shape
        xc      = x.permute(0, 2, 1)                               # (B, C, T)
        patches = xc.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # -> (B, C, num_patches, patch_len)
        _, _, P, PL = patches.shape
        patches = patches.reshape(B * C, P, PL)                    # (B*C, P, patch_len)
        emb     = self.patch_proj(patches) + self.pos_enc          # (B*C, P, d_model)
        return self.drop_emb(emb)

    def encode(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return (B, d_model * n_channels) feature vectors — used for t-SNE."""
        B, T, C = x.shape
        x       = self.revin(x, "norm")
        emb     = self._patch_and_embed(x)                         # (B*C, P, d_model)
        out     = self.transformer(emb)                            # (B*C, P, d_model)
        pooled  = self.norm(out.mean(dim=1))                       # (B*C, d_model)
        return pooled.reshape(B, C * self.d_model)                 # (B, C*d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.head(self.encode(x, mask))
