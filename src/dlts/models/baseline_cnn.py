from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection - match dimensions if needed
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += res
        x = self.gelu(x)
        
        return x


class BaselineCNNClassifier(nn.Module):
    """Deep ResNet-1D baseline for sequence classification."""

    is_foundation_model = False

    def __init__(
        self, input_dim: int, num_classes: int, width: int = 128, dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # A standard ResNet-1D architecture for time series usually has 3 blocks
        # We start by expanding the input to our base width
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        
        # Three residual blocks. We can use dilations to increase receptive field cheaply
        self.res_blocks = nn.Sequential(
            ResidualBlock1D(width, width, kernel_size=5, dilation=1),
            ResidualBlock1D(width, width, kernel_size=5, dilation=2),
            ResidualBlock1D(width, width, kernel_size=5, dilation=4),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, sequence_length, channels)
        # PyTorch Conv1d expects: (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.classifier(x)
