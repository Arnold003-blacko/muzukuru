"""Standalone evaluation of a saved checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from muzukuru.data.dataset import WindowDataset, collate_windows
from muzukuru.data.splits import partition_index
from muzukuru.features.windows import build_window_index
from muzukuru.models import build_model
from muzukuru.train.metrics import (
    compute_metrics,
    false_alarm_rate_per_hour,
    plot_confusion_matrix,
    predict_loader,
    save_metrics_json,
)
from muzukuru.train.train import load_window_corpus


def run_evaluation(
    cfg: dict,
    checkpoint: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = load_window_corpus(cfg["paths"]["windows"])
    win_cfg = cfg["windows"]
    index = build_window_index(
        sequences,
        length=int(win_cfg["length"]),
        stride=int(win_cfg["stride_eval"]),
    )
    parts = partition_index(
        index,
        group_by=cfg["split"].get("group_by", "subject"),
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    if split not in parts:
        raise ValueError(f"Unknown split: {split}")
    ds = WindowDataset(parts[split], augment_cfg=None)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_windows,
    )

    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    feat_dim = int(ckpt.get("feature_dim", sequences[0]["features"].shape[1]))
    model = build_model(cfg, input_dim=feat_dim).to(device)
    model.load_state_dict(ckpt["model"])

    y_true, y_pred, latency_ms = predict_loader(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, cfg["class_names"])
    fps = float(cfg["pose"].get("target_fps", 30))
    far = false_alarm_rate_per_hour(
        y_true,
        y_pred,
        normal_class=0,
        window_seconds=int(win_cfg["length"]) / fps,
        stride_seconds=int(win_cfg["stride_eval"]) / fps,
    )
    metrics["latency_ms_per_window"] = latency_ms
    metrics["false_alarm_rate"] = far
    metrics["split"] = split

    reports = Path(cfg["paths"]["reports"])
    save_metrics_json(metrics, reports / f"{split}_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        cfg["class_names"],
        reports / f"{split}_confusion_matrix.png",
        title=f"{split.title()} Confusion Matrix",
    )
    return metrics
