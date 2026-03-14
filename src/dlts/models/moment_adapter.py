from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from transformers import AutoModel


class MomentAdapterClassifier(nn.Module):
    """MOMENT foundation model adapted for LSST classification."""

    is_foundation_model = True

    def __init__(
        self,
        num_classes: int,
        moment_model_id: str = "HachiML/MOMENT-1-large-embedding-v0.1",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Load the community-supported transformers-compatible MOMENT model
        self.moment_model = AutoModel.from_pretrained(moment_model_id, trust_remote_code=True)
        
        # MOMENT-large provides a 1024-dimensional embedding
        hidden_size = 1024

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def freeze_backbone(self) -> None:
        for p in self.moment_model.parameters():
            p.requires_grad = False

    def unfreeze_last_n_encoder_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        # MOMENT's backbone is a T5Stack accessed via self.moment_model.encoder
        for block in self.moment_model.encoder.block[-n_layers:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, sequence_length, channels)
        # MOMENT expects: (batch_size, channels, sequence_length)
        x_moment = rearrange(x, 'b s c -> b c s').float()

        # The model directly returns an embedding of shape (batch_size, 1024)
        outputs = self.moment_model(x_moment)
        z = outputs.embeddings

        return self.classifier(z)
