from __future__ import annotations

import torch
from chronos import Chronos2Pipeline
from einops import rearrange, einsum
from torch import nn


class CrossChannelAttention(nn.Module):
    """Self-attention across channel embeddings with learnt query pooling."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def forward(self, channel_tokens: torch.Tensor) -> torch.Tensor:
        """(B, C, H) -> (B, H)"""
        h = self.encoder(channel_tokens)

        # Squeeze pool_query from (1, 1, H) to (H,)
        query = self.pool_query.squeeze()

        # Dot product query (H) with each channel token (B, C, H) -> channel scores (B, C)
        attn_scores = einsum(query, h, "h_dim, b c h_dim -> b c")
        attn_weights = torch.softmax(attn_scores / (h.size(-1) ** 0.5), dim=-1)

        # Weighted sum of channel tokens (B, C, H) using weights (B, C) -> (B, H)
        return einsum(attn_weights, h, "b c, b c h_dim -> b h_dim")


class ChronosAdapterClassifier(nn.Module):
    """Chronos-2 encoder adapted for LSST classification with cross-channel attention."""

    is_foundation_model = True

    def __init__(
        self,
        num_classes: int,
        chronos_model_id: str = "amazon/chronos-2",
        device_map: str = "cpu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        pipeline = Chronos2Pipeline.from_pretrained(
            chronos_model_id, device_map=device_map
        )
        self.chronos_model = pipeline.model
        hidden_size = int(self.chronos_model.config.d_model)

        self.channel_attention = CrossChannelAttention(
            hidden_size=hidden_size,
            n_heads=2,
            n_layers=2,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def freeze_backbone(self) -> None:
        for p in self.chronos_model.parameters():
            p.requires_grad = False

    def unfreeze_last_n_encoder_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        for block in self.chronos_model.encoder.block[-n_layers:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_channels = x.shape

        model_device = next(self.chronos_model.parameters()).device

        # Batch all channels into one encode call: (B, T, C) → (B*C, T)
        flat = rearrange(x, "b t c -> (b c) t").float().to(model_device)
        encoder_outputs, _, _, num_ctx_patches = self.chronos_model.encode(flat)
        hidden = encoder_outputs.last_hidden_state  # (B*C, P, H)
        pooled = hidden[:, :num_ctx_patches, :].mean(dim=1)  # (B*C, H)
        z = rearrange(pooled, "(b c) h -> b c h", b=bsz).to(x.device)  # (B, C, H)

        z = self.channel_attention(z)
        return self.classifier(z)
