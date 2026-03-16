"""
Post-hoc ensemble over trained checkpoints.

Each training run writes a sidecar JSON to the checkpoint directory:
    {checkpoint_dir}/{model_name}_run_metrics.json

This script:
  1. Globs all sidecar files in a given directory.
  2. Filters out models below --min_val_f1 (default 0.33).
  3. Computes F1-weighted soft-voting weights:

        w_i = exp(T · F1_val,i) / sum_j exp(T · F1_val,j)

     where T = 1/temperature (default 10.0).
  4. Rebuilds each model from its saved config, loads its checkpoint.
  5. Runs inference on the LSST test set and aggregates probability vectors.
  6. Reports per-model and ensemble metrics.
  7. Saves a per-model F1 comparison chart and a confusion matrix to
     {checkpoint_dir}/ and prints their paths for inline display in Colab.

Usage
-----
    uv run python -m dlts.ensemble --checkpoint_dir checkpoints/

    # custom threshold / temperature
    uv run python -m dlts.ensemble \\
        --checkpoint_dir checkpoints/ \\
        --min_val_f1 0.35 \\
        --temperature 10.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from dlts.data.lsst_ts import LSSTDataset, load_lsst
from dlts.metrics import classification_metrics
from dlts.models.factory import build_model


# ── Helpers ───────────────────────────────────────────────────────────────────


def _exp_softmax_weights(val_f1s: list[float], temperature: float) -> np.ndarray:
    """
    F1-weighted softmax:  w_i = exp(T · F1_val,i) / sum_j exp(T · F1_val,j)
    Numerically stable via max-subtraction before exp.
    """
    s = np.array(val_f1s, dtype=np.float64) * temperature
    s -= s.max()
    w = np.exp(s)
    return w / w.sum()


@torch.no_grad()
def _get_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batches = []
    for x, _ in loader:
        logits = model(x.to(device))
        batches.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def _load_member(record: dict, device: torch.device) -> torch.nn.Module:
    """Rebuild model architecture from saved config and load checkpoint weights."""
    mc = record["model_cfg"]
    dc = record["data_cfg"]

    # Derive input_dim and num_classes from the checkpoint itself so we don't
    # need to reload the dataset just to inspect shapes.
    ckpt = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)

    # Infer num_classes from the final linear layer weight shape.
    # Works for all models: last Linear has shape (num_classes, d_in).
    classifier_keys = [
        k
        for k in ckpt
        if k.endswith(".weight")
        and "classifier" in k
        or k.endswith(".weight")
        and k.split(".")[-2] in ("head", "classifier")
    ]
    # Walk keys in reverse order to find the last linear layer.
    linear_keys = [k for k in ckpt if k.endswith(".weight") and ckpt[k].ndim == 2]
    last_linear_key = linear_keys[-1]
    num_classes = ckpt[last_linear_key].shape[0]

    # input_dim: for patch models use n_channels; for conv models use input channels.
    # Stored in model_cfg if the run used the factory's default parameter names.
    # Fallback: 6 (LSST C=6).
    input_dim = 6  # LSST default

    model = build_model(
        model_name=record["model_name"],
        input_dim=input_dim,
        num_classes=num_classes,
        dropout=mc.get("dropout", 0.1),
        inception_nb_filters=mc.get("inception_nb_filters", 32),
        seq_len=mc.get("seq_len", 36),
        patch_len=mc.get("patch_len", 4),
        stride=mc.get("stride", 4),
        d_model=mc.get("d_model", 64),
        n_heads=mc.get("n_heads", 4),
        n_layers=mc.get("n_layers", 3),
        d_ff=mc.get("d_ff", 256),
        dlo_rank=mc.get("dlo_rank", 8),
        chronos_model_id=mc.get("chronos_model_id", "amazon/chronos-2"),
        device_map=str(device),
    )

    # Foundation models need the backbone loaded before the state dict.
    if getattr(model, "is_foundation_model", False):
        if hasattr(model, "load_chronos"):
            model.load_chronos(device=str(device))
        elif hasattr(model, "load_moment"):
            model.load_moment(device=str(device))

    model.load_state_dict(ckpt, strict=False)
    return model.to(device)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _save_plots(
    y_true: np.ndarray,
    p_ensemble: np.ndarray,
    eligible: list[dict],
    weights: np.ndarray,
    per_model_metrics: list[dict],
    class_labels: list,
    out_dir: Path,
) -> list[Path]:
    """
    Save two figures to out_dir and return their paths:
      1. per-model F1 comparison bar chart
      2. ensemble confusion matrix
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    saved: list[Path] = []

    # 1. Per-model F1 comparison ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(eligible) * 1.4), 5))
    model_names = [r["model_name"] for r in eligible]
    val_f1s = [r["val_macro_f1"] for r in eligible]
    test_f1s = [m["macro_f1"] for m in per_model_metrics]

    x = np.arange(len(eligible))
    w = 0.35
    bars_val = ax.bar(
        x - w / 2, val_f1s, w, label="Val macro-F1", color="steelblue", alpha=0.85
    )
    bars_test = ax.bar(
        x + w / 2, test_f1s, w, label="Test macro-F1", color="coral", alpha=0.85
    )

    for bar in (*bars_val, *bars_test):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Annotate ensemble weight above each model pair
    for i, wt in enumerate(weights):
        ax.text(
            x[i],
            max(val_f1s[i], test_f1s[i]) + 0.02,
            f"w={wt:.3f}",
            ha="center",
            fontsize=8,
            color="dimgray",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Per-Model Val / Test Macro-F1 and Ensemble Weights")
    ax.set_ylim(0, 1.08)
    ax.legend()
    plt.tight_layout()
    p = out_dir / "ensemble_model_comparison.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    # 2. Ensemble confusion matrix ────────────────────────────────────────────
    y_pred = p_ensemble.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    n_cls = len(class_labels)
    fig, ax = plt.subplots(figsize=(max(8, n_cls * 0.8), max(7, n_cls * 0.7)))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title("Ensemble — Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()
    p = out_dir / "ensemble_confusion_matrix.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="F1-weighted soft-voting ensemble")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing *_run_metrics.json sidecar files.",
    )
    parser.add_argument(
        "--min_val_f1",
        type=float,
        default=0.33,
        help="Exclude models with val macro F1 below this threshold.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Softmax temperature T in the weighting formula.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto | cpu | cuda | mps.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Apply dataset-level z-normalisation (should match training).",
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

    # ── Load sidecar files ────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    sidecar_files = sorted(ckpt_dir.glob("*_run_metrics.json"))
    if not sidecar_files:
        raise FileNotFoundError(
            f"No *_run_metrics.json files found in '{ckpt_dir}'. "
            "Run train.py for each model first."
        )

    records = []
    for f in sidecar_files:
        with open(f) as fp:
            r = json.load(fp)
        if r.get("checkpoint") is None or not Path(r["checkpoint"]).exists():
            print(f"  [SKIP] {f.name} — checkpoint file missing")
            continue
        records.append(r)

    # ── Filter by val F1 threshold ────────────────────────────────────
    eligible = [r for r in records if r["val_macro_f1"] >= args.min_val_f1]
    excluded = [r for r in records if r["val_macro_f1"] < args.min_val_f1]

    print(
        f"\nFound {len(records)} run(s), {len(eligible)} eligible "
        f"(val F1 ≥ {args.min_val_f1})."
    )
    for r in excluded:
        print(f"  [EXCLUDED] {r['model_name']:20s}  val_f1={r['val_macro_f1']:.4f}")

    if not eligible:
        raise RuntimeError(
            f"No models passed the val F1 threshold of {args.min_val_f1}. "
            "Lower --min_val_f1 or train more models."
        )

    # ── Data ──────────────────────────────────────────────────────────
    print("\nLoading LSST test set...")
    _, _, X_test, y_test, meta = load_lsst(normalize=args.normalize)
    test_ds = LSSTDataset(X_test, y_test)
    test_ldr = DataLoader(test_ds, batch_size=256, shuffle=False)
    y_np = y_test

    # ── Collect per-model probabilities ──────────────────────────────
    all_probs: list[np.ndarray] = []
    all_val_f1s: list[float] = []
    per_model_metrics: list[dict] = []

    print()
    for r in eligible:
        print(f"Loading  {r['model_name']:20s}  val_f1={r['val_macro_f1']:.4f} ...")
        model = _load_member(r, device)
        probs = _get_probs(model, test_ldr, device)

        m = classification_metrics(y_np, probs)
        print(
            f"  -> test_acc={m['accuracy']:.4f}  "
            f"test_f1={m['macro_f1']:.4f}  "
            f"test_bal_acc={m['balanced_accuracy']:.4f}"
        )

        all_probs.append(probs)
        all_val_f1s.append(r["val_macro_f1"])
        per_model_metrics.append(m)

        # Free GPU memory before loading the next model.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Ensemble ──────────────────────────────────────────────────────
    weights = _exp_softmax_weights(all_val_f1s, temperature=args.temperature)

    print("\n── Ensemble weights ─────────────────────────────────────")
    for r, w in zip(eligible, weights):
        print(
            f"  {r['model_name']:20s}  val_f1={r['val_macro_f1']:.4f}  weight={w:.4f}"
        )

    p_ensemble = sum(w * p for w, p in zip(weights, all_probs))
    # Floating-point accumulation of float32 * float64 can leave row sums
    # marginally off 1.0, causing sklearn's log_loss to warn.
    p_ensemble /= p_ensemble.sum(axis=1, keepdims=True)
    ens_metrics = classification_metrics(y_np, p_ensemble)

    print("\n── Ensemble result ──────────────────────────────────────")
    print(
        f"  Accuracy          : {ens_metrics['accuracy']:.4f}\n"
        f"  Macro F1          : {ens_metrics['macro_f1']:.4f}\n"
        f"  Balanced Accuracy : {ens_metrics['balanced_accuracy']:.4f}\n"
        f"  Log-Loss          : {ens_metrics['log_loss']:.4f}"
    )

    # Per-class F1 breakdown (critical for imbalanced LSST)
    from sklearn.metrics import f1_score

    y_pred = p_ensemble.argmax(axis=1)
    per_class_f1 = f1_score(y_np, y_pred, average=None)
    unique_classes, class_counts = np.unique(y_np, return_counts=True)
    count_map = dict(zip(unique_classes, class_counts))

    print("\n── Per-class F1 (test) ──────────────────────────────────")
    print(f"  {'Class':>8}  {'F1':>6}  {'Support':>8}")
    print(f"  {'-' * 8}  {'-' * 6}  {'-' * 8}")
    for cls_idx, f1 in zip(unique_classes, per_class_f1):
        label = meta.class_labels[cls_idx]
        n = count_map.get(cls_idx, 0)
        print(f"  {str(label):>8}  {f1:.4f}  {n:>8d}")
    print(f"  {'-' * 8}  {'-' * 6}  {'-' * 8}")
    print(f"  {'macro':>8}  {per_class_f1.mean():.4f}  {len(y_np):>8d}")

    # ── Save plots ────────────────────────────────────────────────────
    plot_paths = _save_plots(
        y_true=y_np,
        p_ensemble=p_ensemble,
        eligible=eligible,
        weights=weights,
        per_model_metrics=per_model_metrics,
        class_labels=meta.class_labels,
        out_dir=ckpt_dir,
    )
    print("\nPlots saved:")
    for p in plot_paths:
        print(f"  {p}")
    print("\n# To display in Colab after this cell, run:")
    for p in plot_paths:
        print(f'#   IPython.display.display(IPython.display.Image("{p}"))')

    # ── Save result ───────────────────────────────────────────────────
    result = {
        "members": [
            {
                "model_name": r["model_name"],
                "val_macro_f1": r["val_macro_f1"],
                "weight": float(w),
                "checkpoint": r["checkpoint"],
            }
            for r, w in zip(eligible, weights)
        ],
        "excluded": [
            {"model_name": r["model_name"], "val_macro_f1": r["val_macro_f1"]}
            for r in excluded
        ],
        "ensemble_metrics": ens_metrics,
        "per_class_f1": {
            str(meta.class_labels[cls_idx]): float(f1)
            for cls_idx, f1 in zip(unique_classes, per_class_f1)
        },
        "min_val_f1": args.min_val_f1,
        "temperature": args.temperature,
        "plots": [str(p) for p in plot_paths],
    }
    out_path = ckpt_dir / "ensemble_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nEnsemble result saved -> {out_path}")


if __name__ == "__main__":
    main()
