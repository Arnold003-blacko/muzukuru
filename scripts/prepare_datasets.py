#!/usr/bin/env python3
"""
Download and prepare public fall datasets for Muzukuru training.

Primary: UP-Fall improved 3D skeletons (Zenodo 12773013) — already MediaPipe 33.
Supplement: UR Fall (official RGB → MediaPipe) and optional Roboflow.
Supplement: Le2i (place videos under data/external/le2i, then MediaPipe).

Example:
  python scripts/prepare_datasets.py --upfall --urfall --urfall-falls 10 --urfall-adls 10
  python scripts/prepare_datasets.py --le2i   # after placing Le2i files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzukuru.config import ensure_dirs, load_config
from muzukuru.datasets.labels import CLASS_NAMES
from muzukuru.datasets.le2i import ingest_le2i
from muzukuru.datasets.upfall import download_upfall, extract_zips, ingest_upfall
from muzukuru.datasets.urfall import (
    download_urfall_rgb,
    download_urfall_roboflow,
    ingest_urfall,
)
from muzukuru.datasets.write import save_all, summarize_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--upfall", action="store_true", help="Download+ingest UP-Fall Zenodo")
    parser.add_argument("--urfall", action="store_true", help="Download+ingest UR Fall RGB")
    parser.add_argument("--urfall-falls", type=int, default=5)
    parser.add_argument("--urfall-adls", type=int, default=5)
    parser.add_argument("--urfall-max-clips", type=int, default=None)
    parser.add_argument("--roboflow", action="store_true", help="Also try Roboflow UR Fall")
    parser.add_argument("--le2i", action="store_true", help="Ingest Le2i from data/external/le2i")
    parser.add_argument("--le2i-max-videos", type=int, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="UP-Fall + UR Fall (default subset) + Le2i if present",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to metadata instead of replacing rows for new clip_ids only",
    )
    args = parser.parse_args()

    if args.all:
        args.upfall = True
        args.urfall = True
        args.le2i = True

    if not (args.upfall or args.urfall or args.le2i or args.roboflow):
        parser.error("Specify --upfall and/or --urfall and/or --le2i (or --all)")

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    feat_cfg = cfg["features"]
    pose_cfg = cfg["pose"]

    external = ROOT / "data" / "external"
    up_dir = external / "upfall"
    ur_dir = external / "urfall"
    le_dir = external / "le2i"

    all_clips: list = []

    if args.upfall:
        download_upfall(up_dir)
        extracted = extract_zips(up_dir, up_dir / "extracted")
        all_clips.extend(ingest_upfall(extracted, feat_cfg=feat_cfg))

    if args.roboflow:
        download_urfall_roboflow(ur_dir / "roboflow_dl")

    if args.urfall:
        download_urfall_rgb(
            ur_dir / "zips",
            n_falls=args.urfall_falls,
            n_adls=args.urfall_adls,
        )
        all_clips.extend(
            ingest_urfall(
                ur_dir / "zips",
                ur_dir / "frames",
                feat_cfg=feat_cfg,
                max_clips=args.urfall_max_clips,
                max_frames=300,
            )
        )

    if args.le2i:
        all_clips.extend(
            ingest_le2i(
                le_dir,
                feat_cfg=feat_cfg,
                pose_cfg=pose_cfg,
                max_videos=args.le2i_max_videos,
                max_frames=400,
            )
        )

    if not all_clips:
        print("No clips produced. Check downloads / paths.")
        sys.exit(1)

    poses = Path(cfg["paths"]["poses"])
    windows = Path(cfg["paths"]["windows"])
    meta = Path(cfg["paths"]["metadata"])

    # Drop previous synthetic-only corpus when rebuilding unless appending
    if not args.keep_existing:
        for d in (poses, windows):
            for p in d.glob("*"):
                if p.name.startswith(".") or p.name == ".gitkeep":
                    continue
                p.unlink()

    df = save_all(
        all_clips,
        poses,
        windows,
        meta,
        append=args.keep_existing,
        min_window=int(cfg["windows"]["length"]),
    )
    counts = summarize_labels(all_clips)
    print("\n=== Label counts (clips) ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"\nClasses: {CLASS_NAMES}")
    print(f"Metadata: {meta}  ({len(df)} rows)")
    print(f"Windows:  {windows}")
    print("\nNext: python scripts/train.py")


if __name__ == "__main__":
    main()
