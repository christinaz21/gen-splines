"""
memory.py — EMA-based persistent memory anchors.

All variants implement the same (update, get_anchor) interface but operate
on different primitive types: spline control points (N, K, 3), point cloud
positions (P, 3), or Gaussian splat means (P, 3). The behavior is identical;
the class hierarchy just makes call sites self-documenting.
"""

import torch


class _EMAMemory:
    def __init__(self, initial_params: torch.Tensor, ema_decay: float = 0.8):
        self.anchor = initial_params.clone().detach()
        self.ema_decay = ema_decay

    def update(self, new_params: torch.Tensor) -> None:
        self.anchor = (
            self.ema_decay * self.anchor
            + (1.0 - self.ema_decay) * new_params.detach()
        )

    def get_anchor(self) -> torch.Tensor:
        return self.anchor.clone()


class PersistentCurveMemory(_EMAMemory):
    """For spline control points (N, K, 3)."""


class PersistentPointMemory(_EMAMemory):
    """For point cloud positions (P, 3)."""


class PersistentGaussianMemory(_EMAMemory):
    """For Gaussian splat means (P, 3). Scale/opacity tracked separately."""
