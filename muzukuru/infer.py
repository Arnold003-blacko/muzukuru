"""Run the trained BiGRU model on a video or precomputed feature sequence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from muzukuru.config import load_config
from muzukuru.features.engineering import compute_frame_features
from muzukuru.features.windows import iter_windows
from muzukuru.models import build_model
from muzukuru.pose.extract import extract_pose_sequence


def _torch_load(path: str | Path, map_location) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_classifier(
    checkpoint: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict, torch.device]:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint or Path(cfg["paths"]["checkpoints"]) / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")

    ckpt = _torch_load(ckpt_path, map_location=device)
    feat_dim = int(ckpt.get("feature_dim", 114))
    model = build_model(cfg, input_dim=feat_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, device


@torch.no_grad()
def classify_features(
    model: torch.nn.Module,
    features: np.ndarray,
    valid: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Classify sliding windows; returns list of prediction dicts."""
    length = int(cfg["windows"]["length"])
    stride = int(cfg["windows"].get("stride_eval", length))
    class_names = cfg["class_names"]
    results: list[dict[str, Any]] = []

    for start, window in iter_windows(
        features, length=length, stride=stride, valid=valid, min_valid_ratio=0.4
    ):
        x = torch.from_numpy(window).unsqueeze(0).to(device)
        logits, attn = model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_id = int(probs.argmax())
        results.append(
            {
                "start_frame": start,
                "end_frame": start + length,
                "label": class_names[pred_id],
                "label_id": pred_id,
                "confidence": float(probs[pred_id]),
                "probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
                "attention": None
                if attn is None
                else attn[0].cpu().numpy().tolist(),
            }
        )
    return results


def classify_video(
    video_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    config_path: str | Path | None = None,
    max_frames: int | None = 300,
) -> dict[str, Any]:
    model, cfg, device = load_classifier(checkpoint, config_path)
    pose_cfg = cfg["pose"]
    pose = extract_pose_sequence(
        video_path,
        model_complexity=int(pose_cfg.get("model_complexity", 1)),
        min_detection_confidence=float(pose_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.5)),
        max_frames=max_frames,
    )
    feat_cfg = cfg["features"]
    features, feat_meta = compute_frame_features(
        pose["landmarks"],
        pose["valid"],
        include_visibility=bool(feat_cfg.get("include_visibility", False)),
        include_acceleration=bool(feat_cfg.get("include_acceleration", True)),
        include_torso_angle=bool(feat_cfg.get("include_torso_angle", True)),
        include_bbox_aspect=bool(feat_cfg.get("include_bbox_aspect", True)),
        include_com_height=bool(feat_cfg.get("include_com_height", True)),
    )
    windows = classify_features(model, features, pose["valid"], cfg, device)
    return {
        "fps": pose["fps"],
        "frame_count": pose["frame_count"],
        "pose_valid_ratio": float(pose["valid"].mean()) if len(pose["valid"]) else 0.0,
        "feature_dim": feat_meta["feature_dim"],
        "windows": windows,
        "summary_label": _majority_label(windows, cfg["class_names"]),
    }


def classify_npz_clip(
    npz_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    model, cfg, device = load_classifier(checkpoint, config_path)
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    valid = data["valid"].astype(bool) if "valid" in data.files else np.ones(len(features), bool)
    windows = classify_features(model, features, valid, cfg, device)
    return {
        "clip_id": str(data["clip_id"]) if "clip_id" in data.files else Path(npz_path).stem,
        "true_label": int(data["label"]) if "label" in data.files else None,
        "frame_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "windows": windows,
        "summary_label": _majority_label(windows, cfg["class_names"]),
    }


def _majority_label(windows: list[dict], class_names: list[str]) -> str:
    if not windows:
        return "unknown"
    # Prefer highest-confidence non-normal if any danger class appears
    danger = {"fall", "collapse", "prolonged_immobility", "abnormal_repetitive_movement"}
    danger_hits = [w for w in windows if w["label"] in danger]
    if danger_hits:
        return max(danger_hits, key=lambda w: w["confidence"])["label"]
    counts = {n: 0 for n in class_names}
    for w in windows:
        counts[w["label"]] += 1
    return max(counts, key=counts.get)
