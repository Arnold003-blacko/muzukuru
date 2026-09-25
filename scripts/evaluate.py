#!/usr/bin/env python3
"""Evaluate a checkpoint; writes metrics + confusion matrix under reports/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzukuru.config import ensure_dirs, load_config
from muzukuru.train.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to checkpoints/best.pt",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    ckpt = args.checkpoint or str(Path(cfg["paths"]["checkpoints"]) / "best.pt")
    metrics = run_evaluation(cfg, ckpt, split=args.split)
    print(json.dumps(
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "latency_ms_per_window": metrics["latency_ms_per_window"],
            "false_alarms_per_hour": metrics["false_alarm_rate"]["false_alarms_per_hour"],
            "per_class": metrics["per_class"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
