from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from tslearn.datasets import UCR_UEA_datasets


@dataclass(frozen=True)
class TSMetadata:
    n_dimensions: int
    series_length: int
    class_labels: list[int]


def _fill_nan(X: np.ndarray) -> np.ndarray:
    """Forward-fill, backward-fill, then zero-fill NaN along the time axis."""
    X = X.copy()
    for i in range(X.shape[0]):
        for c in range(X.shape[2]):
            col = X[i, :, c]
            mask = np.isnan(col)
            if not mask.any():
                continue
            idx = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col = col[idx]
            mask = np.isnan(col)
            idx = np.where(~mask, np.arange(len(col)), len(col) - 1)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            col = col[idx]
            X[i, :, c] = np.nan_to_num(col, nan=0.0)
    return X


def _normalize_dataset(
    X_train: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Global per-channel z-normalization fit on training data."""
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    scale = np.maximum(std, eps)

    return (X_train - mean) / scale, (X_test - mean) / scale


def _encode_labels(
    y_train_raw: np.ndarray, y_test_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Encodes string labels into integers [0, C-1] fit on training data."""
    labels = sorted({int(v) for v in y_train_raw})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    y_train = np.array([label_to_idx[int(v)] for v in y_train_raw], dtype=np.int64)
    y_test = np.array([label_to_idx[int(v)] for v in y_test_raw], dtype=np.int64)

    return y_train, y_test, labels


def load_lsst(
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TSMetadata]:
    ds = UCR_UEA_datasets()
    X_train, y_train_raw, X_test, y_test_raw = ds.load_dataset("LSST")

    X_train = _fill_nan(X_train)
    X_test = _fill_nan(X_test)

    y_train, y_test, labels = _encode_labels(y_train_raw, y_test_raw)

    if normalize:
        X_train, X_test = _normalize_dataset(X_train, X_test)
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    metadata = TSMetadata(
        n_dimensions=int(X_train.shape[-1]),
        series_length=int(X_train.shape[1]),
        class_labels=labels,
    )
    return X_train, y_train, X_test, y_test, metadata


class LSSTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, X: np.ndarray, y: np.ndarray, device: torch.device | None = None
    ) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
