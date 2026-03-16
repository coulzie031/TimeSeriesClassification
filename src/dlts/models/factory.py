from __future__ import annotations

from torch import nn

from dlts.models.chronos_adapter import ChronosAdapterClassifier
from dlts.models.inception_time import InceptionTime
from dlts.models.moment_adapter import MomentAdapterClassifier
from dlts.models.patch_tst import PatchTSTClassifier
from dlts.models.units import UniTSClassifier


def build_model(
    model_name: str,
    # shared
    input_dim: int,
    num_classes: int,
    dropout: float,
    # cnn models
    inception_nb_filters: int,
    # patch-based models (PatchTST, UniTS)
    seq_len: int,
    patch_len: int,
    stride: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dlo_rank: int,
    # foundation models
    chronos_model_id: str,
    device_map: str,
) -> nn.Module:
    if model_name == "inception_time":
        return InceptionTime(
            input_dim=input_dim,
            num_classes=num_classes,
            nb_filters=inception_nb_filters,
            dropout=dropout,
        )
    if model_name == "patch_tst":
        return PatchTSTClassifier(
            seq_len=seq_len,
            n_channels=input_dim,
            num_classes=num_classes,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
    if model_name == "units":
        return UniTSClassifier(
            seq_len=seq_len,
            n_channels=input_dim,
            num_classes=num_classes,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            dlo_rank=dlo_rank,
        )
    if model_name == "chronos":
        return ChronosAdapterClassifier(
            num_classes=num_classes,
            chronos_model_id=chronos_model_id,
            device_map=device_map,
            dropout=dropout,
        )
    if model_name == "moment":
        return MomentAdapterClassifier(
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_name={model_name!r}")
