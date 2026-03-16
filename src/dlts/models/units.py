"""
UniTS — Unified Multi-Task Time Series Model, adapted for classification.

Reference: Gao et al., NeurIPS 2024 — https://arxiv.org/abs/2403.00131

Key design decisions vs the mhani/ re-implementation
------------------------------------------------------
1. Variable attention operates correctly over the *channel* dimension (C), not over
   timesteps.  In each block, patches are pooled per channel to (B, C, d), MHA is
   applied over the C channel vectors, and the resulting correction is broadcast back
   to every patch of that channel.

2. The [TASK] token is a single global token per sample.  After the encoder blocks,
   per-channel patch representations are GAP-pooled to (B, C, d), the task token is
   prepended -> (B, C+1, d), and one self-attention step lets it aggregate information
   from all channels before the classification head reads from position 0.

3. Patch-based input (consistent with PatchTST) rather than timestep-projection,
   giving the model better inductive bias for T=36 LSST sequences.

Architecture
------------
  (B, T, C)
    │
    ├─ patch + embed (channel-independent)  ->  (B*C, P, d)
    │
    ├─ N × UniTSBlock
    │     ├─ sequence attention   (B*C, P, d)   — temporal, within each channel
    │     ├─ variable attention   (B, C, d)     — cross-channel, pooled then broadcast
    │     ├─ DLO                  (B*C, P, d)   — dense low-rank time mixing
    │     └─ FFN                  (B*C, P, d)
    │
    ├─ GAP over patches  ->  (B, C, d)
    ├─ prepend [TASK] token  ->  (B, C+1, d)
    ├─ one self-attention step
    └─ head( task_token )  ->  logits (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DynamicLinearOperator(nn.Module):
    """
    Dense low-rank time-domain mixing (DLO).

    Learns W = U @ V ∈ R^{P×P} and applies it along the patch/time dimension,
    giving the model a non-attention pathway to mix temporal information.
    Low-rank factorisation (rank ≪ P) keeps parameter count small.
    """

    def __init__(self, num_patches: int, rank: int = 8) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.empty(num_patches, rank))
        self.V = nn.Parameter(torch.empty(rank, num_patches))
        nn.init.trunc_normal_(self.U, std=0.02)
        nn.init.trunc_normal_(self.V, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*C, P, d)
        W = self.U @ self.V  # (P, P)
        return (x.transpose(1, 2) @ W.T).transpose(1, 2)  # (B*C, P, d)


class UniTSBlock(nn.Module):
    """
    One UniTS transformer block.

    Processing order
    ----------------
    1. Sequence attention  — standard MHA over patches within each channel
    2. Variable attention  — MHA over channels (correctly over C, not T):
                             pool (B*C, P, d) -> (B, C, d),
                             attend -> (B, C, d),
                             broadcast residual back to all patches.
    3. DLO                 — dense low-rank time mixing
    4. FFN
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_patches: int,
        dropout: float = 0.1,
        dlo_rank: int = 8,
    ) -> None:
        super().__init__()

        self.seq_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.var_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dlo = DynamicLinearOperator(num_patches, rank=dlo_rank)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, B: int, C: int) -> Tensor:
        # x: (B*C, P, d)
        # Pre-LN: x = x + f(norm(x)) — residual stream stays unnormalised,
        # preventing activation blow-up across blocks.
        P, d = x.shape[1], x.shape[2]

        # ── 1. Sequence attention (temporal, within each channel) ─────────────
        xn = self.norm1(x)
        h, _ = self.seq_attn(xn, xn, xn)
        x = x + self.dropout(h)

        # ── 2. Variable attention (cross-channel) ─────────────────────────────
        # Pool normed patches -> one representative vector per channel: (B, C, d)
        channel_repr = self.norm2(x).reshape(B, C, P, d).mean(dim=2)
        h, _ = self.var_attn(channel_repr, channel_repr, channel_repr)
        # Broadcast the cross-channel correction back to every patch
        x = x + self.dropout(h).reshape(B * C, 1, d)

        # ── 3. DLO (dense time mixing) ────────────────────────────────────────
        h = self.dlo(self.norm3(x))
        x = x + self.dropout(h)

        # ── 4. FFN ────────────────────────────────────────────────────────────
        x = x + self.ffn(self.norm4(x))

        return x


class UniTSClassifier(nn.Module):
    """
    UniTS adapted for LSST multivariate time series classification.

    Differences from a standard PatchTST (which it subsumes):
      • Variable attention  — cross-channel MHA in every block
      • DLO                 — non-attention dense time-domain mixing in every block
      • [TASK] token        — global learnable token aggregates all channel information
                              before the classification head, rather than plain GAP

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
        dlo_rank: int = 8,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch projection and positional encoding (shared across channels)
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        self.drop_emb = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                UniTSBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    num_patches=self.num_patches,
                    dropout=dropout,
                    dlo_rank=dlo_rank,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm_enc = nn.LayerNorm(d_model)

        # Global [TASK] token: prepended after encoder, attends over all C channels
        self.task_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.task_token, std=0.02)
        self.task_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_task = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Dropout(dropout_head),
            nn.Linear(d_model, num_classes),
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
        xc = x.permute(0, 2, 1)  # (B, C, T)
        patches = xc.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # -> (B, C, num_patches, patch_len)
        _, _, P, PL = patches.shape
        patches = patches.reshape(B * C, P, PL)  # (B*C, P, patch_len)
        emb = self.patch_proj(patches) + self.pos_enc  # (B*C, P, d_model)
        return self.drop_emb(emb)

    def encode(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return (B, d_model) task-token representations — used for t-SNE."""
        B, T, C = x.shape

        h = self._patch_and_embed(x)  # (B*C, P, d_model)

        for block in self.blocks:
            h = block(h, B, C)

        # GAP over patches -> one vector per channel
        h = self.norm_enc(h.mean(dim=1))  # (B*C, d_model)
        channels = h.reshape(B, C, self.d_model)  # (B, C, d_model)

        # Prepend [TASK] token, let it attend over all channel representations
        task = self.task_token.expand(B, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([task, channels], dim=1)  # (B, C+1, d_model)
        seq_n = self.norm_task(seq)
        h2, _ = self.task_attn(seq_n, seq_n, seq_n)
        seq = seq + h2                             # Pre-LN residual

        return seq[:, 0]  # (B, d_model) — task token

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.head(self.encode(x, mask))
