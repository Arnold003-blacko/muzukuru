#!/usr/bin/env python3
"""Build per-clip feature sequences from extracted pose .npz files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzukuru.config import ensure_dirs, load_config
from muzukuru.features.engineering import compute_frame_features
from muzukuru.pose.extract import load_pose_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    pose_dir = Path(cfg["paths"]["poses"])
    out_dir = Path(cfg["paths"]["windows"])
    feat_cfg = cfg["features"]

    meta_files = sorted(pose_dir.glob("*.meta.json"))
    if not meta_files:
        raise SystemExit(
            f"No pose metadata in {pose_dir}. Run scripts/extract_poses.py first."
        )

    for meta_path in tqdm(meta_files, desc="features"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        clip_id = meta["clip_id"]
        pose = load_pose_npz(pose_dir / f"{clip_id}.npz")
        features, feat_meta = compute_frame_features(
            pose["landmarks"],
            pose["valid"],
            include_visibility=bool(feat_cfg.get("include_visibility", False)),
            include_acceleration=bool(feat_cfg.get("include_acceleration", True)),
            include_torso_angle=bool(feat_cfg.get("include_torso_angle", True)),
            include_bbox_aspect=bool(feat_cfg.get("include_bbox_aspect", True)),
            include_com_height=bool(feat_cfg.get("include_com_height", True)),
        )
        out_path = out_dir / f"{clip_id}.npz"
        np.savez_compressed(
            out_path,
            features=features,
            valid=pose["valid"],
            label=np.array(meta["label"], dtype=np.int64),
            subject_id=np.array(meta["subject_id"]),
            session_id=np.array(meta["session_id"]),
            clip_id=np.array(clip_id),
            feature_dim=np.array(feat_meta["feature_dim"], dtype=np.int32),
            layout=np.array(json.dumps(feat_meta["layout"])),
        )

    print(f"Feature sequences written to {out_dir}")


if __name__ == "__main__":
    main()
