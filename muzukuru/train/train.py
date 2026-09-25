"""Training loop: AdamW, class-weighted CE / focal loss, cosine or plateau LR."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from muzukuru.data.dataset import WindowDataset, collate_windows
from muzukuru.data.splits import (
    class_counts,
    inverse_frequency_weights,
    partition_index,
    summarize_split,
)
from muzukuru.features.engineering import feature_dim_from_cfg
from muzukuru.models import build_model
from muzukuru.train.metrics import (
    compute_metrics,
    false_alarm_rate_per_hour,
    plot_confusion_matrix,
    predict_loader,
    save_metrics_json,
)


def _torch_load(path: str | Path, map_location) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_criterion(
    cfg: dict,
    class_weight: torch.Tensor | None,
    device: torch.device,
) -> nn.Module:
    w = class_weight.to(device) if class_weight is not None else None
    if cfg["train"].get("loss", "weighted_ce") == "focal":
        return FocalLoss(weight=w, gamma=float(cfg["train"].get("focal_gamma", 2.0)))
    return nn.CrossEntropyLoss(weight=w)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    name = cfg["train"].get("scheduler", "cosine")
    epochs = int(cfg["train"]["epochs"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    return None


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * len(y)
        correct += int((logits.argmax(-1) == y).sum().item())
        n += len(y)
    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0
    correct = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        correct += int((logits.argmax(-1) == y).sum().item())
        n += len(y)
    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}


def load_window_corpus(windows_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load per-clip .npz files produced by scripts/build_windows.py.

    Expected keys: features, valid, label, subject_id, session_id, clip_id
    """
    windows_dir = Path(windows_dir)
    sequences: list[dict[str, Any]] = []
    for path in sorted(windows_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        sequences.append(
            {
                "features": data["features"].astype(np.float32),
                "valid": data["valid"].astype(bool),
                "label": int(data["label"]),
                "subject_id": str(data["subject_id"]),
                "session_id": str(data["session_id"]),
                "clip_id": str(data["clip_id"]),
            }
        )
    return sequences


def run_training(cfg: dict, *, pretrained_ckpt: str | Path | None = None) -> dict[str, Any]:
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences = load_window_corpus(cfg["paths"]["windows"])
    if not sequences:
        raise FileNotFoundError(
            f"No window .npz files in {cfg['paths']['windows']}. "
            "Run extract_poses.py and build_windows.py first."
        )

    from muzukuru.features.windows import build_window_index

    win_cfg = cfg["windows"]
    full_index = build_window_index(
        sequences,
        length=int(win_cfg["length"]),
        stride=int(win_cfg["stride_train"]),
    )
    if not full_index:
        raise RuntimeError("Window index is empty; check window length vs clip lengths.")

    parts = partition_index(
        full_index,
        group_by=cfg["split"].get("group_by", "subject"),
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    split_summary = summarize_split(parts)

    # Rebuild val/test with non-overlapping stride for cleaner metrics
    eval_index = build_window_index(
        sequences,
        length=int(win_cfg["length"]),
        stride=int(win_cfg["stride_eval"]),
    )
    eval_parts = partition_index(
        eval_index,
        group_by=cfg["split"].get("group_by", "subject"),
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        seed=int(cfg["seed"]),
    )

    counts = class_counts(parts["train"], int(cfg["num_classes"]))
    weights = inverse_frequency_weights(counts)
    class_weight = torch.tensor(weights, dtype=torch.float32)

    feat_dim = int(sequences[0]["features"].shape[1])
    expected = feature_dim_from_cfg(cfg["features"])
    if feat_dim != expected:
        # Prefer actual data dim
        pass

    include_vis = bool(cfg["features"].get("include_visibility", False))
    train_ds = WindowDataset(
        parts["train"],
        augment_cfg=cfg.get("augmentation"),
        include_visibility=include_vis,
        seed=int(cfg["seed"]),
    )
    val_ds = WindowDataset(eval_parts["val"], augment_cfg=None)
    test_ds = WindowDataset(eval_parts["test"], augment_cfg=None)

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_windows
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_windows
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_windows
    )

    model = build_model(cfg, input_dim=feat_dim).to(device)

    if pretrained_ckpt:
        ckpt = _torch_load(pretrained_ckpt, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))
        if freeze_epochs > 0:
            model.freeze_backbone(freeze_early_gru=True)

    criterion = build_criterion(cfg, class_weight, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(len(train_loader), 1))

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_path = ckpt_dir / "best.pt"
    patience = int(cfg["train"].get("early_stopping_patience", 10))
    stale = 0
    history: list[dict[str, Any]] = []

    epochs = int(cfg["train"]["epochs"])
    freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))

    for epoch in range(1, epochs + 1):
        if pretrained_ckpt and freeze_epochs and epoch == freeze_epochs + 1:
            model.unfreeze_all()
            optimizer = build_optimizer(model, cfg)

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats["acc"])
            else:
                scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}  "
            f"train_loss={row['train_loss']:.4f} acc={row['train_acc']:.3f}  "
            f"val_loss={row['val_loss']:.4f} acc={row['val_acc']:.3f}"
        )

        if val_stats["acc"] > best_val:
            best_val = val_stats["acc"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "feature_dim": feat_dim,
                    "class_names": cfg["class_names"],
                    "epoch": epoch,
                    "val_acc": best_val,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test evaluation with best checkpoint
    ckpt = _torch_load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    y_true, y_pred, latency_ms = predict_loader(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, cfg["class_names"])
    window_s = int(win_cfg["length"]) / float(cfg["pose"].get("target_fps", 30))
    stride_s = int(win_cfg["stride_eval"]) / float(cfg["pose"].get("target_fps", 30))
    far = false_alarm_rate_per_hour(
        y_true, y_pred, normal_class=0, window_seconds=window_s, stride_seconds=stride_s
    )
    metrics["latency_ms_per_window"] = latency_ms
    metrics["false_alarm_rate"] = far
    metrics["split_summary"] = split_summary
    metrics["class_weights"] = weights.tolist()

    save_metrics_json(metrics, reports / "test_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        cfg["class_names"],
        reports / "confusion_matrix.png",
        title="Test Confusion Matrix",
    )
    with open(reports / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best checkpoint: {best_path}")
    print(f"Test macro-F1: {metrics['macro_f1']:.4f}  acc: {metrics['accuracy']:.4f}")
    print(f"Latency: {latency_ms:.2f} ms/window")
    print(f"False alarms/hour (normal): {far['false_alarms_per_hour']:.3f}")
    return metrics
