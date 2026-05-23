"""Exponential moving average of model weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters(), strict=True):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.ema_model.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.ema_model.load_state_dict(state)
