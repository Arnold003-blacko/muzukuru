"""UR Fall Detection Dataset ingest (official RGB + optional Roboflow)."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from muzukuru.datasets.labels import URFALL_NAME_MAP, to_id
from muzukuru.features.engineering import compute_frame_features
from muzukuru.pose.extract import extract_pose_sequence

# Official host (RGB frame zips)
UR_BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data"


def download_urfall_rgb(
    dest: Path,
    *,
    n_falls: int = 30,
    n_adls: int = 20,
    camera: str = "cam0",
) -> Path:
    """Download a subset of official UR Fall RGB frame archives."""
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    jobs: list[str] = []
    for i in range(1, n_falls + 1):
        jobs.append(f"fall-{i:02d}-{camera}-rgb.zip")
    for i in range(1, n_adls + 1):
        jobs.append(f"adl-{i:02d}-{camera}-rgb.zip")

    for name in jobs:
        out = dest / name
        if out.exists() and out.stat().st_size > 100_000:
            print(f"[urfall] exists {name} ({out.stat().st_size // 1_000_000} MB)")
            continue
        if out.exists() and out.stat().st_size < 100_000:
            out.unlink()
        url = f"{UR_BASE}/{name}"
        print(f"[urfall] downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, out)
            if out.stat().st_size < 100_000:
                print(f"[urfall] incomplete {name} ({out.stat().st_size} bytes); removing")
                out.unlink(missing_ok=True)
        except Exception as e:
            print(f"[urfall] FAILED {name}: {e}")
            out.unlink(missing_ok=True)
    return dest


def download_urfall_roboflow(
    dest: Path,
    *,
    workspace: str = "fall-detection-w7nxl",
    project: str = "ur-fall",
    version: int = 1,
    api_key: str | None = None,
) -> Path | None:
    """
    Download from Roboflow Universe if ROBOFLOW_API_KEY is set.

    The slug 'ur-fall-detection-dataset-mbbxe' is not always public; default
    falls back to the well-known 'ur-fall' project. Override via env if needed.
    """
    import os

    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("[urfall] ROBOFLOW_API_KEY not set; skipping Roboflow download.")
        return None
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[urfall] pip install roboflow to enable Roboflow download.")
        return None

    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("folder", location=str(dest / "roboflow"))
    print(f"[urfall] Roboflow dataset at {dataset}")
    return Path(dest / "roboflow")


def _frames_from_rgb_zip(zip_path: Path, extract_root: Path) -> Path:
    out = extract_root / zip_path.stem
    if out.exists() and any(out.rglob("*.png")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    return out


def _sorted_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)


def extract_pose_from_image_dir(
    image_dir: Path,
    *,
    model_complexity: int = 1,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run MediaPipe Pose over an ordered image sequence."""
    import mediapipe as mp

    from muzukuru.config import NUM_LANDMARKS
    from muzukuru.pose.extract import _landmark_array

    images = _sorted_images(image_dir)
    if max_frames is not None:
        images = images[:max_frames]
    frames: list[np.ndarray] = []
    valid: list[bool] = []

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                frames.append(np.zeros((NUM_LANDMARKS, 4), np.float32))
                valid.append(False)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks is not None:
                frames.append(_landmark_array(result.pose_landmarks))
                valid.append(True)
            else:
                frames.append(np.zeros((NUM_LANDMARKS, 4), np.float32))
                valid.append(False)

    landmarks = (
        np.stack(frames, 0)
        if frames
        else np.zeros((0, 33, 4), np.float32)
    )
    return {
        "landmarks": landmarks,
        "valid": np.asarray(valid, dtype=bool),
        "fps": 30.0,
        "frame_count": len(frames),
    }


def _label_from_name(name: str) -> str:
    lower = name.lower()
    for key, mapped in URFALL_NAME_MAP.items():
        if key in lower:
            return mapped
    if lower.startswith("fall") or "-fall" in lower:
        return "fall"
    return "normal_activity"


def convert_urfall_zip(
    zip_path: Path,
    extract_root: Path,
    *,
    feat_cfg: dict,
    pose_complexity: int = 1,
    max_frames: int | None = 300,
) -> dict[str, Any] | None:
    folder = _frames_from_rgb_zip(zip_path, extract_root)
    pose = extract_pose_from_image_dir(
        folder, model_complexity=pose_complexity, max_frames=max_frames
    )
    if pose["frame_count"] < 30:
        return None
    label_name = _label_from_name(zip_path.stem)
    features, fmeta = compute_frame_features(
        pose["landmarks"],
        pose["valid"],
        include_visibility=bool(feat_cfg.get("include_visibility", False)),
        include_acceleration=bool(feat_cfg.get("include_acceleration", True)),
        include_torso_angle=bool(feat_cfg.get("include_torso_angle", True)),
        include_bbox_aspect=bool(feat_cfg.get("include_bbox_aspect", True)),
        include_com_height=bool(feat_cfg.get("include_com_height", True)),
    )
    m = re.search(r"(fall|adl)-(\d+)", zip_path.stem, re.I)
    subj = f"urfall_{m.group(1)}_{m.group(2)}" if m else f"urfall_{zip_path.stem}"
    return {
        "landmarks": pose["landmarks"],
        "valid": pose["valid"],
        "features": features,
        "feature_dim": fmeta["feature_dim"],
        "layout": fmeta["layout"],
        "label": to_id(label_name),
        "label_name": label_name,
        "subject_id": subj,
        "session_id": subj,
        "clip_id": f"urfall_{zip_path.stem}",
        "source": "urfall",
    }


def ingest_urfall(
    zip_dir: Path,
    extract_root: Path,
    *,
    feat_cfg: dict,
    max_clips: int | None = None,
    max_frames: int | None = 300,
) -> list[dict[str, Any]]:
    zips = sorted(zip_dir.glob("*-rgb.zip"))
    if max_clips is not None:
        zips = zips[:max_clips]
    clips: list[dict[str, Any]] = []
    for zp in zips:
        print(f"[urfall] pose extract {zp.name}")
        try:
            clip = convert_urfall_zip(
                zp, extract_root, feat_cfg=feat_cfg, max_frames=max_frames
            )
        except Exception as e:
            print(f"[urfall] skip {zp.name}: {e}")
            continue
        if clip is not None:
            clips.append(clip)
    print(f"[urfall] converted {len(clips)} clips")
    return clips
