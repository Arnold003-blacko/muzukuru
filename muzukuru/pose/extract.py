"""MediaPipe Pose extraction from video files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from muzukuru.config import NUM_LANDMARKS


def _landmark_array(landmarks) -> np.ndarray:
    """Return (33, 4) float32 array: x, y, z, visibility."""
    out = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        out[i] = (lm.x, lm.y, lm.z, lm.visibility)
    return out


def extract_pose_sequence(
    video_path: str | Path,
    *,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """
    Run MediaPipe Pose on every frame.

    Returns dict with:
      landmarks: (T, 33, 4) float32
      valid: (T,) bool — True when a pose was detected
      fps: float
      frame_count: int
      width, height: int
    """
    import cv2
    import mediapipe as mp

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    valid: list[bool] = []

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and idx >= max_frames:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks is not None:
                frames.append(_landmark_array(result.pose_landmarks))
                valid.append(True)
            else:
                frames.append(np.zeros((NUM_LANDMARKS, 4), dtype=np.float32))
                valid.append(False)
            idx += 1

    cap.release()
    landmarks = (
        np.stack(frames, axis=0)
        if frames
        else np.zeros((0, NUM_LANDMARKS, 4), dtype=np.float32)
    )
    return {
        "landmarks": landmarks,
        "valid": np.asarray(valid, dtype=bool),
        "fps": fps,
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "video_path": str(video_path),
    }


def save_pose_npz(path: str | Path, pose_data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        landmarks=pose_data["landmarks"],
        valid=pose_data["valid"],
        fps=np.array(pose_data["fps"], dtype=np.float32),
        frame_count=np.array(pose_data["frame_count"], dtype=np.int32),
        width=np.array(pose_data["width"], dtype=np.int32),
        height=np.array(pose_data["height"], dtype=np.int32),
    )


def load_pose_npz(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {
        "landmarks": data["landmarks"].astype(np.float32),
        "valid": data["valid"].astype(bool),
        "fps": float(data["fps"]),
        "frame_count": int(data["frame_count"]),
        "width": int(data["width"]),
        "height": int(data["height"]),
    }
