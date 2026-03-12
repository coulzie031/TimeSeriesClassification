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


def _normalize_dataset(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Vectorized robust channel z-score normalization over the entire dataset."""
    X = X.astype(np.float32)
    # X shape is (batch, seq_len, channels). We normalize over seq_len (axis=1)
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True)
    scale = np.maximum(mad * 1.4826, eps)
    return (X - med) / scale



def _encode_labels(
    y_train_raw: np.ndarray, y_test_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    labels = sorted({int(v) for v in np.concatenate([y_train_raw, y_test_raw], axis=0)})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_train = np.array([label_to_idx[int(v)] for v in y_train_raw], dtype=np.int64)
    y_test = np.array([label_to_idx[int(v)] for v in y_test_raw], dtype=np.int64)
    return y_train, y_test, labels


def load_lsst(
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TSMetadata]:
    ds = UCR_UEA_datasets()
    X_train, y_train_raw, X_test, y_test_raw = ds.load_dataset("LSST")
    if X_train is None or X_test is None:
        raise RuntimeError("tslearn could not load LSST dataset.")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train, y_test, labels = _encode_labels(y_train_raw, y_test_raw)

    if normalize:
        X_train = _normalize_dataset(X_train)
        X_test = _normalize_dataset(X_test)

    metadata = TSMetadata(
        n_dimensions=int(X_train.shape[-1]),
        series_length=int(X_train.shape[1]),
        class_labels=labels,
    )
    return X_train, y_train, X_test, y_test, metadata


class LSSTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: np.ndarray, y: np.ndarray, device: torch.device | None = None) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
