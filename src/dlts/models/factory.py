from __future__ import annotations

from torch import nn

from dlts.models.baseline_cnn import BaselineCNNClassifier
from dlts.models.chronos_adapter import ChronosAdapterClassifier
from dlts.models.moment_adapter import MomentAdapterClassifier


def build_model(
    model_name: str,
    input_dim: int,
    num_classes: int,
    dropout: float,
    baseline_width: int,
    chronos_model_id: str,
    device_map: str,
) -> nn.Module:
    if model_name == "baseline_cnn":
        return BaselineCNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            width=baseline_width,
            dropout=dropout,
        )
    if model_name == "chronos_tsfm":
        return ChronosAdapterClassifier(
            num_classes=num_classes,
            chronos_model_id=chronos_model_id,
            device_map=device_map,
            dropout=dropout,
        )
    if model_name == "moment_tsfm":
        return MomentAdapterClassifier(
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_name={model_name!r}")
