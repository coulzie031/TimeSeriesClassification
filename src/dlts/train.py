from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from jsonargparse import ArgumentParser
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader

from dlts.data.lsst_ts import LSSTDataset, load_lsst
from dlts.losses import inverse_frequency_class_weights
from dlts.metrics import classification_metrics
from dlts.models.factory import build_model


@dataclass
class StageConfig:
    epochs: int
    lr: float
    weight_decay: float


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="LSST TSFM adaptation training")
    parser.add_argument("--cfg", action="config")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    parser.add_argument("--data.normalize", type=bool, default=True)
    parser.add_argument("--data.val_fraction", type=float, default=0.2)
    parser.add_argument("--data.augment", type=bool, default=True)

    parser.add_argument(
        "--model.name",
        type=str,
        default="inception_time",
        choices=["inception_time", "patch_tst", "units", "chronos", "moment"],
    )
    parser.add_argument("--model.dropout", type=float, default=0.2)
    parser.add_argument("--model.unfreeze_last_n", type=int, default=2)
    # CNN models
    parser.add_argument("--model.inception_nb_filters", type=int, default=32)
    # Patch-based models (PatchTST, UniTS)
    parser.add_argument("--model.seq_len", type=int, default=36)
    parser.add_argument("--model.patch_len", type=int, default=4)
    parser.add_argument("--model.stride", type=int, default=4)
    parser.add_argument("--model.d_model", type=int, default=64)
    parser.add_argument("--model.n_heads", type=int, default=4)
    parser.add_argument("--model.n_layers", type=int, default=3)
    parser.add_argument("--model.d_ff", type=int, default=256)
    parser.add_argument("--model.dlo_rank", type=int, default=8)
    # Foundation models
    parser.add_argument(
        "--model.chronos_model_id", type=str, default="amazon/chronos-2"
    )
    parser.add_argument("--model.device_map", type=str, default=None)

    parser.add_argument("--loss.label_smoothing", type=float, default=0.1)

    # Stage 1: head only (frozen backbone)
    parser.add_argument("--stage1.epochs", type=int, default=10)
    parser.add_argument("--stage1.lr", type=float, default=1e-3)
    parser.add_argument("--stage1.weight_decay", type=float, default=1e-4)

    # Stage 2: discriminative fine-tuning (last N layers unfrozen)
    parser.add_argument("--stage2.epochs", type=int, default=20)
    parser.add_argument("--stage2.lr", type=float, default=1e-5)
    parser.add_argument("--stage2.weight_decay", type=float, default=1e-4)
    parser.add_argument("--stage2.head_lr_scale", type=float, default=10.0)

    parser.add_argument("--wandb.project", type=str, default="lsst-tsfm")
    parser.add_argument("--wandb.entity", type=str, default=None)
    parser.add_argument("--wandb.run_name", type=str, default=None)
    parser.add_argument(
        "--wandb.mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
    )

    return parser


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    device_str = device_arg.lower()

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_str == "cpu":
        return torch.device("cpu")

    if device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError("Requested device 'mps' but MPS is not available.")

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{device_arg}' but CUDA is not available."
            )
        parsed_device = torch.device(device_arg)
        if (
            parsed_device.index is not None
            and parsed_device.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                f"Requested device '{device_arg}' but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        return parsed_device

    raise ValueError(f"Unsupported device string: {device_arg!r}")


def make_optimizer(
    model: nn.Module,
    stage: StageConfig,
    head_lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    head_params = (
        list(model.classifier.parameters()) if hasattr(model, "classifier") else []
    )
    backbone_params = [
        p
        for _, p in model.named_parameters()
        if p.requires_grad and all(hp is not p for hp in head_params)
    ]
    if head_params:
        param_groups: list[dict] = [
            {
                "params": backbone_params,
                "lr": stage.lr,
                "weight_decay": stage.weight_decay,
            },
            {
                "params": [p for p in head_params if p.requires_grad],
                "lr": stage.lr * head_lr_scale,
                "weight_decay": stage.weight_decay,
            },
        ]
    else:
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": stage.lr,
                "weight_decay": stage.weight_decay,
            },
        ]
    return torch.optim.AdamW(param_groups)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing with a short linear warmup."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: np.ndarray | None = None,
) -> dict[str, float]:
    model.eval()
    all_probs = []
    all_targets = []
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, enabled=device.type in ["cuda", "mps"]),
    ):
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    probs_np = np.concatenate(all_probs, axis=0)
    y_np = np.concatenate(all_targets, axis=0)
    return classification_metrics(y_np, probs_np, class_weights=class_weights)


def run_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    stage_name: str,
    stage_cfg: StageConfig,
    head_lr_scale: float = 1.0,
    grad_clip: float = 1.0,
    patience: int = 5,
    checkpoint_path: Path | None = None,
    class_weights: np.ndarray | None = None,
) -> tuple[float, bool]:
    """Returns (best val macro_f1, checkpoint_saved_for_this_stage)."""
    optimizer = make_optimizer(model, stage_cfg, head_lr_scale=head_lr_scale)
    scheduler = make_scheduler(
        optimizer,
        num_epochs=stage_cfg.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=1,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type in ["cuda", "mps"])
    best_f1 = 0.0
    epochs_without_improvement = 0
    saved_checkpoint = False
    model.train()
    for epoch in range(stage_cfg.epochs):
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, enabled=device.type in ["cuda", "mps"]
            ):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device, class_weights=class_weights)
        log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
        log_dict[f"{stage_name}/train_loss"] = float(np.mean(losses))
        log_dict[f"{stage_name}/lr"] = scheduler.get_last_lr()[0]
        log_dict["epoch"] = epoch + 1
        wandb.log(log_dict)

        macro_f1 = val_metrics["macro_f1"]
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                saved_checkpoint = True
        else:
            epochs_without_improvement += 1

        print(
            f"[{stage_name}] epoch={epoch + 1}/{stage_cfg.epochs} "
            f"loss={log_dict[f'{stage_name}/train_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={macro_f1:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_logloss={val_metrics['log_loss']:.4f} "
            f"best_val_f1={best_f1:.4f}"
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"[{stage_name}] Early stopping: no improvement for {patience} epochs."
            )
            break

        model.train()

    wandb.run.summary[f"{stage_name}/best_val_macro_f1"] = best_f1
    return best_f1, saved_checkpoint


def main() -> None:
    parser = build_parser()
    cfg = parser.parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # ── Data loading ──────────────────────────────────────────────────
    X_all_train, y_all_train, X_test, y_test, meta = load_lsst(
        normalize=cfg.data.normalize
    )

    # ── Stratified train / val split ──────────────────────────────────
    val_frac = cfg.data.val_fraction
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=cfg.seed)
    train_idx, val_idx = next(sss.split(X_all_train, y_all_train))

    X_train, y_train = X_all_train[train_idx], y_all_train[train_idx]
    X_val, y_val = X_all_train[val_idx], y_all_train[val_idx]
    print(
        f"Split: {len(train_idx)} train / {len(val_idx)} val / "
        f"{len(y_test)} test  (val_fraction={val_frac})"
    )

    train_ds = LSSTDataset(X_train, y_train, device=device, augment=cfg.data.augment)
    val_ds   = LSSTDataset(X_val,   y_val,   device=device, augment=False)
    test_ds  = LSSTDataset(X_test,  y_test,  device=device, augment=False)
    # Data tensors are preloaded onto `device`, so worker processes are only useful on CPU.
    effective_num_workers = cfg.num_workers if device.type == "cpu" else 0
    loader_kwargs = dict(
        num_workers=effective_num_workers,
        pin_memory=False,  # Not needed if data is already on device
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # ── Model ─────────────────────────────────────────────────────────
    num_classes = len(meta.class_labels)
    input_dim = X_train.shape[-1]
    model_device_map = cfg.model.device_map or device.type
    model = build_model(
        model_name=cfg.model.name,
        input_dim=input_dim,
        num_classes=num_classes,
        dropout=cfg.model.dropout,
        inception_nb_filters=cfg.model.inception_nb_filters,
        seq_len=cfg.model.seq_len,
        patch_len=cfg.model.patch_len,
        stride=cfg.model.stride,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dlo_rank=cfg.model.dlo_rank,
        chronos_model_id=cfg.model.chronos_model_id,
        device_map=model_device_map,
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────
    y_tensor = torch.from_numpy(y_train)
    class_weights = inverse_frequency_class_weights(
        y_tensor, num_classes=num_classes
    ).to(device)
    class_weights_np = class_weights.cpu().numpy()

    criterion: nn.Module = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg.loss.label_smoothing
    )

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        config=cfg.as_dict(),
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    stage1_ckpt = ckpt_dir / f"{cfg.model.name}_stage1_best.pt"
    stage2_ckpt = ckpt_dir / f"{cfg.model.name}_stage2_best.pt"
    stage1_saved = False
    stage2_saved = False

    if getattr(model, "is_foundation_model", False):
        print(f"\n>>> [FOUNDATION] Adapting {cfg.model.name} via 2-Stage LP-FT")

        # ── Stage 1: Linear Probing ──────────────────────────────────
        print(">>> Stage 1: Freezing backbone for head-only warmup...")
        model.freeze_backbone()
        _, stage1_saved = run_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            stage_name="stage1",
            stage_cfg=StageConfig(
                epochs=cfg.stage1.epochs,
                lr=cfg.stage1.lr,
                weight_decay=cfg.stage1.weight_decay,
            ),
            head_lr_scale=1.0,
            grad_clip=cfg.grad_clip,
            patience=cfg.early_stopping_patience,
            checkpoint_path=stage1_ckpt,
            class_weights=class_weights_np,
        )

        # ── Stage 2: Fine-Tuning ─────────────────────────────────────
        print(f">>> Stage 2: Unfreezing last {cfg.model.unfreeze_last_n} layers...")
        model.unfreeze_last_n_encoder_layers(cfg.model.unfreeze_last_n)
        best_val_f1, stage2_saved = run_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            stage_name="stage2",
            stage_cfg=StageConfig(
                epochs=cfg.stage2.epochs,
                lr=cfg.stage2.lr,
                weight_decay=cfg.stage2.weight_decay,
            ),
            head_lr_scale=cfg.stage2.head_lr_scale,
            grad_clip=cfg.grad_clip,
            patience=cfg.early_stopping_patience,
            checkpoint_path=stage2_ckpt,
            class_weights=class_weights_np,
        )
    else:
        print(f"\n>>> [SCRATCH] Training {cfg.model.name} end-to-end")
        best_val_f1, stage1_saved = run_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            stage_name="train",
            stage_cfg=StageConfig(
                epochs=cfg.stage1.epochs,
                lr=cfg.stage1.lr,
                weight_decay=cfg.stage1.weight_decay,
            ),
            head_lr_scale=1.0,
            grad_clip=cfg.grad_clip,
            patience=cfg.early_stopping_patience,
            checkpoint_path=stage1_ckpt,
            class_weights=class_weights_np,
        )

    # ── Final eval on held-out test set ────────────────────────────────
    if stage2_saved and stage2_ckpt.exists():
        model.load_state_dict(torch.load(stage2_ckpt, map_location=device))
        print(f"Loaded best Stage 2 checkpoint (val_f1={best_val_f1:.4f})")
    elif stage1_saved and stage1_ckpt.exists():
        model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print(f"Loaded best checkpoint (val_f1={best_val_f1:.4f})")
    else:
        print(
            "No checkpoint from this run was saved; evaluating current model weights."
        )

    test_metrics = evaluate(model, test_loader, device, class_weights=class_weights_np)
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    print(
        f"Test  | acc={test_metrics['accuracy']:.4f} "
        f"f1={test_metrics['macro_f1']:.4f} "
        f"bal_acc={test_metrics['balanced_accuracy']:.4f} "
        f"logloss={test_metrics['log_loss']:.4f}"
    )

    # ── Save sidecar JSON for ensemble ────────────────────────────────
    # val_macro_f1 is used (not test) for ensemble weighting to avoid leakage.
    best_ckpt = (
        stage2_ckpt if (stage2_saved and stage2_ckpt.exists()) else
        stage1_ckpt if (stage1_saved and stage1_ckpt.exists()) else
        None
    )
    run_record = {
        "model_name":            cfg.model.name,
        "run_name":              cfg.wandb.run_name,
        "checkpoint":            str(best_ckpt) if best_ckpt else None,
        "val_macro_f1":          best_val_f1,
        "test_macro_f1":         test_metrics["macro_f1"],
        "test_accuracy":         test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "model_cfg":             {k: v for k, v in cfg.as_dict().get("model", {}).items()},
        "data_cfg":              {k: v for k, v in cfg.as_dict().get("data", {}).items()},
    }
    sidecar_path = ckpt_dir / f"{cfg.model.name}_run_metrics.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"Run metrics saved -> {sidecar_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
