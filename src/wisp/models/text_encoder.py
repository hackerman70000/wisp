"""Frozen CLIP text encoder used as the conditioning signal."""

from __future__ import annotations

from functools import lru_cache

import open_clip
import torch
from torch import Tensor, nn


@lru_cache(maxsize=2)
def _load_clip(arch: str, pretrained: str, device: str):
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


class CLIPTextEncoder(nn.Module):
    """Wraps an open_clip text encoder and projects its embedding to ``out_dim``."""

    def __init__(self, arch: str, pretrained: str, out_dim: int) -> None:
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.out_dim = out_dim
        clip_model, tokenizer = _load_clip(arch, pretrained, "cpu")
        clip_device = next(clip_model.parameters()).device
        with torch.no_grad():
            dummy = tokenizer(["x"]).to(clip_device)
            in_dim = clip_model.encode_text(dummy).shape[-1]
        self._clip = clip_model
        self._tokenizer = tokenizer
        self.projection = nn.Linear(in_dim, out_dim)

    @torch.no_grad()
    def _embed(self, prompts: list[str]) -> Tensor:
        device = self.projection.weight.device
        if next(self._clip.parameters()).device != device:
            self._clip.to(device)
        tokens = self._tokenizer(prompts).to(device)
        return self._clip.encode_text(tokens).float()

    def forward(self, prompts: list[str]) -> Tensor:
        feats = self._embed(prompts)
        return self.projection(feats)
