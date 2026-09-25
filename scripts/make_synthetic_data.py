#!/usr/bin/env python3
"""
Generate synthetic pose/feature sequences for a smoke-test of the training stack.

Does NOT replace real video data. Use this to verify install + train loop before
recording or downloading fall datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzukuru.config import NUM_LANDMARKS, ensure_dirs, load_config
from muzukuru.features.engineering import compute_frame_features
from muzukuru.pose.extract import save_pose_npz  # needs numpy only via extract's save


def _synthetic_landmarks(t: int, label: int, rng: np.random.Generator) -> np.ndarray:
    """Crude motion templates so classes are linearly separable-ish."""
    lm = np.zeros((t, NUM_LANDMARKS, 4), dtype=np.float32)
    # Base standing pose
    for i in range(NUM_LANDMARKS):
        lm[:, i, 0] = 0.5 + 0.02 * np.sin(np.linspace(0, 4, t) + i * 0.1)
        lm[:, i, 1] = 0.3 + 0.4 * (i / NUM_LANDMARKS)
        lm[:, i, 2] = 0.0
        lm[:, i, 3] = 0.9

    # Hips / shoulders roughly
    lm[:, 23, 1] = 0.55
    lm[:, 24, 1] = 0.55
    lm[:, 11, 1] = 0.35
    lm[:, 12, 1] = 0.35
    lm[:, 0, 1] = 0.25

    tt = np.linspace(0, 1, t)
    if label == 1:  # fall: rapid downward hip + torso flatten
        drop = np.clip((tt - 0.3) / 0.25, 0, 1)
        lm[:, :, 1] += 0.35 * drop[:, None]
        lm[:, 11:13, 1] += 0.15 * drop[:, None]
    elif label == 2:  # collapse: similar but slower + more lateral
        drop = np.clip((tt - 0.2) / 0.5, 0, 1)
        lm[:, :, 1] += 0.3 * drop[:, None]
        lm[:, :, 0] += 0.1 * drop[:, None]
    elif label == 3:  # prolonged immobility: flat on ground, tiny noise
        lm[:, :, 1] = 0.75 + 0.01 * rng.normal(size=(t, NUM_LANDMARKS))
    elif label == 4:  # abnormal repetitive: oscillating arms/torso
        osc = 0.08 * np.sin(2 * np.pi * 5 * tt)
        lm[:, 11:17, 0] += osc[:, None]
    else:  # normal: gentle sway
        lm[:, :, 0] += 0.02 * np.sin(2 * np.pi * tt)[:, None]

    lm += rng.normal(0, 0.005, size=lm.shape).astype(np.float32)
    lm[:, :, 3] = np.clip(lm[:, :, 3], 0.5, 1.0)
    return lm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--clips-per-class", type=int, default=2)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    rng = np.random.default_rng(args.seed)

    pose_dir = Path(cfg["paths"]["poses"])
    win_dir = Path(cfg["paths"]["windows"])
    class_names = cfg["class_names"]
    feat_cfg = cfg["features"]

    rows = []
    clip_n = 0
    for s in range(args.subjects):
        subject = f"subj{s:02d}"
        for label, name in enumerate(class_names):
            for c in range(args.clips_per_class):
                clip_id = f"{subject}_{name}_{c}"
                session_id = f"{subject}_sess0"
                lm = _synthetic_landmarks(args.frames, label, rng)
                valid = np.ones(args.frames, dtype=bool)
                save_pose_npz(
                    pose_dir / f"{clip_id}.npz",
                    {
                        "landmarks": lm,
                        "valid": valid,
                        "fps": 30.0,
                        "frame_count": args.frames,
                        "width": 640,
                        "height": 480,
                    },
                )
                (pose_dir / f"{clip_id}.meta.json").write_text(
                    json.dumps(
                        {
                            "clip_id": clip_id,
                            "subject_id": subject,
                            "session_id": session_id,
                            "label": label,
                            "video_path": f"synthetic/{clip_id}.mp4",
                            "fps": 30.0,
                            "frame_count": args.frames,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                features, feat_meta = compute_frame_features(
                    lm,
                    valid,
                    include_visibility=bool(feat_cfg.get("include_visibility", False)),
                    include_acceleration=bool(feat_cfg.get("include_acceleration", True)),
                    include_torso_angle=bool(feat_cfg.get("include_torso_angle", True)),
                    include_bbox_aspect=bool(feat_cfg.get("include_bbox_aspect", True)),
                    include_com_height=bool(feat_cfg.get("include_com_height", True)),
                )
                np.savez_compressed(
                    win_dir / f"{clip_id}.npz",
                    features=features,
                    valid=valid,
                    label=np.array(label, dtype=np.int64),
                    subject_id=np.array(subject),
                    session_id=np.array(session_id),
                    clip_id=np.array(clip_id),
                    feature_dim=np.array(feat_meta["feature_dim"], dtype=np.int32),
                    layout=np.array(json.dumps(feat_meta["layout"])),
                )
                rows.append(
                    {
                        "video_path": f"synthetic/{clip_id}.mp4",
                        "clip_id": clip_id,
                        "subject_id": subject,
                        "session_id": session_id,
                        "label": name,
                    }
                )
                clip_n += 1

    import pandas as pd

    meta_path = Path(cfg["paths"]["metadata"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    print(f"Generated {clip_n} synthetic clips -> {win_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Feature dim: {feat_meta['feature_dim']}  layout: {feat_meta['layout']}")


if __name__ == "__main__":
    main()
