"""Evaluation metrics for imbalanced multi-class fall classification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return y_true, y_pred, mean latency_ms per batch-item (window)."""
    model.eval()
    ys, preds = [], []
    latencies = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        t0 = time.perf_counter()
        logits, _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        pred = logits.argmax(dim=-1).cpu().numpy()
        ys.append(y)
        preds.append(pred)
        latencies.append(elapsed / max(len(y), 1) * 1000.0)
    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    return y_true, y_pred, mean_latency


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "per_class": {
                n: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for n in class_names
            },
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "accuracy": 0.0,
            "confusion_matrix": [[0] * len(class_names) for _ in class_names],
            "classification_report": {},
            "empty": True,
        }
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    per_class = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    return {
        "per_class": per_class,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]) if "accuracy" in report else float(
            (y_true == y_pred).mean() if len(y_true) else 0.0
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def false_alarm_rate_per_hour(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normal_class: int = 0,
    window_seconds: float = 3.0,
    stride_seconds: float | None = None,
) -> dict[str, float]:
    """
    Among windows that are truly normal, rate of non-normal predictions,
    extrapolated to alarms per hour of normal footage.

    Uses non-overlapping assumption if stride_seconds is None (= window_seconds).
    """
    stride = window_seconds if stride_seconds is None else stride_seconds
    mask = y_true == normal_class
    n_normal = int(mask.sum())
    if n_normal == 0:
        return {"false_alarms": 0.0, "normal_windows": 0.0, "false_alarms_per_hour": 0.0}
    false_alarms = int(((y_pred[mask] != normal_class)).sum())
    hours = (n_normal * stride) / 3600.0
    rate = false_alarms / hours if hours > 0 else 0.0
    return {
        "false_alarms": float(false_alarms),
        "normal_windows": float(n_normal),
        "false_alarms_per_hour": float(rate),
    }


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_names: list[str],
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
