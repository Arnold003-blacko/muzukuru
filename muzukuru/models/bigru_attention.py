"""Bidirectional GRU with temporal attention for windowed fall classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Additive attention over time steps; returns context vector and weights."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, H)
        logits = self.score(h).squeeze(-1)  # (B, T)
        weights = torch.softmax(logits, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return context, weights


class BiGRUAttentionClassifier(nn.Module):
    """
    1D-CNN → Bidirectional GRU (2 layers) → Attention → FC → Softmax logits.

    Input: (batch, T, F)
    Output: logits (batch, num_classes), attention weights (batch, T)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        cnn_channels: int = 64,
        cnn_kernel: int = 3,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        fc_hidden: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.gru_layers = gru_layers

        pad = cnn_kernel // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=cnn_kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(cnn_channels),
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        rnn_out = gru_hidden * (2 if bidirectional else 1)
        self.attention = TemporalAttention(rnn_out) if use_attention else None

        self.head = nn.Sequential(
            nn.Linear(rnn_out, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: (B, T, F) → CNN expects (B, F, T)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        h, _ = self.gru(h)  # (B, T, H*)

        attn_w = None
        if self.use_attention and self.attention is not None:
            context, attn_w = self.attention(h)
        else:
            context = h.mean(dim=1)

        logits = self.head(context)
        return logits, attn_w

    def freeze_backbone(self, freeze_early_gru: bool = True) -> None:
        """Freeze CNN and early GRU layers for transfer fine-tuning."""
        for p in self.cnn.parameters():
            p.requires_grad = False
        if freeze_early_gru and self.gru_layers > 1:
            # GRU parameter names: weight_ih_l0, weight_hh_l0, ... and l0_reverse
            for name, p in self.gru.named_parameters():
                if "_l0" in name:
                    p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def build_model(cfg: dict, input_dim: int) -> BiGRUAttentionClassifier:
    m = cfg["model"]
    return BiGRUAttentionClassifier(
        input_dim=input_dim,
        num_classes=int(cfg["num_classes"]),
        cnn_channels=int(m["cnn_channels"]),
        cnn_kernel=int(m["cnn_kernel"]),
        gru_hidden=int(m["gru_hidden"]),
        gru_layers=int(m["gru_layers"]),
        fc_hidden=int(m["fc_hidden"]),
        dropout=float(m["dropout"]),
        bidirectional=bool(m.get("bidirectional", True)),
        use_attention=bool(m.get("use_attention", True)),
    )
