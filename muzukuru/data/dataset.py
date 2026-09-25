"""PyTorch Dataset for windowed pose feature sequences."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from muzukuru.features.augment import augment_window


class WindowDataset(Dataset):
    def __init__(
        self,
        index: list[dict[str, Any]],
        *,
        augment_cfg: dict[str, Any] | None = None,
        include_visibility: bool = False,
        seed: int = 42,
    ) -> None:
        self.index = index
        self.augment_cfg = augment_cfg
        self.include_visibility = include_visibility
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = self.index[i]
        start = int(item["start"])
        length = int(item["length"])
        feats = item["features"][start : start + length].astype(np.float32)

        if self.augment_cfg is not None:
            # Per-sample seed derived from base + index for reproducibility across epochs
            # if Dataset is recreated; within epoch still stochastic via rng state.
            feats = augment_window(
                feats,
                self.augment_cfg,
                self.rng,
                include_visibility=self.include_visibility,
            )

        x = torch.from_numpy(feats)
        y = torch.tensor(int(item["label"]), dtype=torch.long)
        return {"x": x, "y": y, "start": torch.tensor(start)}


def collate_windows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    xs = torch.stack([b["x"] for b in batch], dim=0)
    ys = torch.stack([b["y"] for b in batch], dim=0)
    return {"x": xs, "y": ys}
