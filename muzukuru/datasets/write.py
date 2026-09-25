"""Write ingested clips to data/poses + data/windows and rebuild metadata.csv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from muzukuru.datasets.labels import CLASS_NAMES
from muzukuru.pose.extract import save_pose_npz


def pad_sequence(features: np.ndarray, valid: np.ndarray, min_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad by repeating the last frame so short clips still yield one window."""
    t = features.shape[0]
    if t >= min_len:
        return features, valid
    if t == 0:
        return features, valid
    pad_n = min_len - t
    features = np.concatenate([features, np.repeat(features[-1:], pad_n, axis=0)], axis=0)
    valid = np.concatenate([valid, np.repeat(valid[-1:], pad_n, axis=0)], axis=0)
    return features.astype(np.float32), valid.astype(bool)


def save_clip(clip: dict[str, Any], poses_dir: Path, windows_dir: Path, *, min_window: int = 60) -> dict[str, str]:
    poses_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)
    clip_id = clip["clip_id"]

    features = clip["features"].astype(np.float32)
    valid = clip["valid"].astype(bool)
    features, valid = pad_sequence(features, valid, min_window)

    save_pose_npz(
        poses_dir / f"{clip_id}.npz",
        {
            "landmarks": clip["landmarks"],
            "valid": clip["valid"],
            "fps": float(clip.get("fps", 30.0)),
            "frame_count": int(clip["landmarks"].shape[0]),
            "width": int(clip.get("width", 0)),
            "height": int(clip.get("height", 0)),
        },
    )
    (poses_dir / f"{clip_id}.meta.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "subject_id": clip["subject_id"],
                "session_id": clip["session_id"],
                "label": int(clip["label"]),
                "label_name": clip.get("label_name", CLASS_NAMES[int(clip["label"])]),
                "source": clip.get("source", "unknown"),
                "fps": float(clip.get("fps", 30.0)),
                "frame_count": int(clip["landmarks"].shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        windows_dir / f"{clip_id}.npz",
        features=features,
        valid=valid,
        label=np.array(clip["label"], dtype=np.int64),
        subject_id=np.array(clip["subject_id"]),
        session_id=np.array(clip["session_id"]),
        clip_id=np.array(clip_id),
        feature_dim=np.array(clip["feature_dim"], dtype=np.int32),
        layout=np.array(json.dumps(clip.get("layout", []))),
        source=np.array(clip.get("source", "unknown")),
    )
    return {
        "video_path": f"external/{clip.get('source', 'unknown')}/{clip_id}",
        "clip_id": clip_id,
        "subject_id": clip["subject_id"],
        "session_id": clip["session_id"],
        "label": clip.get("label_name", CLASS_NAMES[int(clip["label"])]),
        "source": clip.get("source", "unknown"),
    }


def save_all(
    clips: list[dict[str, Any]],
    poses_dir: Path,
    windows_dir: Path,
    metadata_path: Path,
    *,
    append: bool = False,
    min_window: int = 60,
) -> pd.DataFrame:
    rows = []
    for clip in clips:
        rows.append(save_clip(clip, poses_dir, windows_dir, min_window=min_window))
    df_new = pd.DataFrame(rows)
    if append and metadata_path.exists():
        df_old = pd.read_csv(metadata_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["clip_id"], keep="last")
    else:
        df = df_new
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)
    return df


def summarize_labels(clips: list[dict[str, Any]]) -> dict[str, int]:
    counts = {n: 0 for n in CLASS_NAMES}
    for c in clips:
        name = c.get("label_name", CLASS_NAMES[int(c["label"])])
        counts[name] = counts.get(name, 0) + 1
    return counts
