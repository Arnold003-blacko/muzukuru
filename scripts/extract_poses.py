#!/usr/bin/env python3
"""Extract MediaPipe Pose landmarks from labeled videos listed in metadata.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzukuru.config import ensure_dirs, load_config
from muzukuru.pose.extract import extract_pose_sequence, save_pose_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument(
        "--metadata",
        default=None,
        help="CSV with columns: video_path,clip_id,subject_id,session_id,label",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    meta_path = Path(args.metadata or cfg["paths"]["metadata"])
    if not meta_path.exists():
        raise SystemExit(
            f"Metadata not found: {meta_path}\n"
            "Create data/metadata.csv (see data/metadata.example.csv)."
        )

    df = pd.read_csv(meta_path)
    required = {"video_path", "clip_id", "subject_id", "session_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"metadata.csv missing columns: {sorted(missing)}")

    pose_cfg = cfg["pose"]
    out_dir = Path(cfg["paths"]["poses"])
    label_map = {name: i for i, name in enumerate(cfg["class_names"])}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        video = Path(str(row["video_path"]))
        if not video.is_absolute():
            video = ROOT / video
        clip_id = str(row["clip_id"])
        out_path = out_dir / f"{clip_id}.npz"
        if out_path.exists():
            continue

        pose = extract_pose_sequence(
            video,
            model_complexity=int(pose_cfg.get("model_complexity", 1)),
            min_detection_confidence=float(pose_cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.5)),
            max_frames=args.max_frames,
        )
        # Attach label metadata into a sidecar sidecar json? Keep in build step via CSV.
        save_pose_npz(out_path, pose)

        # Normalize string labels to int ids in a small companion file
        label = row["label"]
        if isinstance(label, str) and label in label_map:
            label_id = label_map[label]
        else:
            label_id = int(label)
        meta_out = out_dir / f"{clip_id}.meta.json"
        meta_out.write_text(
            __import__("json").dumps(
                {
                    "clip_id": clip_id,
                    "subject_id": str(row["subject_id"]),
                    "session_id": str(row["session_id"]),
                    "label": label_id,
                    "video_path": str(video),
                    "fps": pose["fps"],
                    "frame_count": pose["frame_count"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Poses written to {out_dir}")


if __name__ == "__main__":
    main()
