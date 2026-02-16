"""Neural network definitions for the steganalysis classifiers."""

from __future__ import annotations

import torch.nn as nn

from .features import FEATURE_COUNT


class SteganalysisDNN(nn.Module):
    """Fully connected classifier over the LSB feature vector.

    The binary and multi-class models share this architecture and differ
    only in the width of the output layer, so a single definition covers
    both checkpoints.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):  # noqa: D102 - trivial delegation
        return self.net(x)
