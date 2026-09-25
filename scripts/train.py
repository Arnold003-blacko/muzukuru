#!/usr/bin/env python3
"""Train the BiGRU+attention temporal classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzukuru.config import ensure_dirs, load_config
from muzukuru.train.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional checkpoint for transfer fine-tuning "
        "(freeze CNN + early GRU for freeze_backbone_epochs).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    run_training(cfg, pretrained_ckpt=args.pretrained)


if __name__ == "__main__":
    main()
