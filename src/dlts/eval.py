"""
Per-checkpoint evaluation script.

Reads a sidecar JSON written by train.py, rebuilds the model, runs inference
on the LSST test set, and reports metrics + saves a confusion matrix.

Usage
-----
    uv run python -m dlts.eval --checkpoint_dir checkpoints/ --model patch_tst

    # auto-detect model if only one sidecar exists in the directory
    uv run python -m dlts.eval --checkpoint_dir checkpoints/

    # To display the confusion matrix in Colab after this cell:
    #   import IPython.display
    #   IPython.display.display(IPython.display.Image("checkpoints/patch_tst_confusion_matrix.png"))
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dlts.data.lsst_ts import LSSTDataset, load_lsst
from dlts.ensemble import _get_probs, _load_member
from dlts.metrics import classification_metrics


def _find_sidecar(checkpoint_dir: Path, model: str | None) -> Path:
    """Locate the sidecar JSON for the requested model."""
    if model is not None:
        p = checkpoint_dir / f"{model}_run_metrics.json"
        if not p.exists():
            raise FileNotFoundError(
                f"No sidecar found at '{p}'. "
                "Run train.py for this model first."
            )
        return p

    sidecars = sorted(checkpoint_dir.glob("*_run_metrics.json"))
    if not sidecars:
        raise FileNotFoundError(
            f"No *_run_metrics.json files found in '{checkpoint_dir}'."
        )
    if len(sidecars) > 1:
        names = [s.stem.replace("_run_metrics", "") for s in sidecars]
        raise ValueError(
            f"Multiple models found in '{checkpoint_dir}': {names}. "
            "Specify one with --model."
        )
    return sidecars[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-checkpoint LSST evaluation")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing the *_run_metrics.json sidecar file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g. patch_tst). Auto-detected if only one sidecar exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto | cpu | cuda | mps.",
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Sidecar ───────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    sidecar = _find_sidecar(ckpt_dir, args.model)
    with open(sidecar) as f:
        record = json.load(f)

    model_name = record["model_name"]
    val_f1 = record["val_macro_f1"]
    print(f"\nModel  : {model_name}")
    print(f"Val F1 : {val_f1:.4f}  (from training run)")

    if record.get("checkpoint") is None or not Path(record["checkpoint"]).exists():
        raise FileNotFoundError(
            f"Checkpoint file missing: {record.get('checkpoint')}. "
            "Re-run train.py for this model."
        )

    # ── Data ──────────────────────────────────────────────────────────
    print("\nLoading LSST test set...")
    _, _, X_test, y_test, meta = load_lsst(normalize=True)
    test_ds = LSSTDataset(X_test, y_test)
    test_ldr = DataLoader(test_ds, batch_size=256, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {record['checkpoint']}")
    model = _load_member(record, device)

    # ── Inference ─────────────────────────────────────────────────────
    probs = _get_probs(model, test_ldr, device)
    probs /= probs.sum(axis=1, keepdims=True)  # renormalise for numerical safety

    # ── Aggregate metrics ─────────────────────────────────────────────
    metrics = classification_metrics(y_test, probs)

    print("\n── Test metrics ─────────────────────────────────────────")
    print(
        f"  Accuracy          : {metrics['accuracy']:.4f}\n"
        f"  Macro F1          : {metrics['macro_f1']:.4f}\n"
        f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}\n"
        f"  Log-Loss          : {metrics['log_loss']:.4f}"
    )

    # ── Per-class F1 ──────────────────────────────────────────────────
    y_pred = probs.argmax(axis=1)
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    unique_classes, class_counts = np.unique(y_test, return_counts=True)
    count_map = dict(zip(unique_classes, class_counts))

    print("\n── Per-class F1 ─────────────────────────────────────────")
    print(f"  {'Class':>8}  {'F1':>6}  {'Support':>8}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*8}")
    for cls_idx, f1 in zip(unique_classes, per_class_f1):
        label = meta.class_labels[cls_idx]
        n = count_map.get(cls_idx, 0)
        print(f"  {str(label):>8}  {f1:.4f}  {n:>8d}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*8}")
    print(f"  {'macro':>8}  {per_class_f1.mean():.4f}  {len(y_test):>8d}")

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    n_cls = len(meta.class_labels)
    fig, ax = plt.subplots(figsize=(max(8, n_cls * 0.8), max(7, n_cls * 0.7)))
    disp = ConfusionMatrixDisplay(cm, display_labels=meta.class_labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(f"{model_name} — Confusion Matrix (Test Set)  |  Macro F1={metrics['macro_f1']:.4f}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()

    plot_path = ckpt_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    print(f"\nConfusion matrix saved -> {plot_path}")
    print("\n# To display in Colab after this cell, run:")
    print(f'#   import IPython.display')
    print(f'#   IPython.display.display(IPython.display.Image("{plot_path}"))')


if __name__ == "__main__":
    main()
