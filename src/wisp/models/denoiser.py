"""Transformer-encoder motion denoiser (MDM-style).

Input  : noisy motion ``x_t`` of shape (B, T, F) where F = J*6 + 3.
Output : predicted clean motion ``x̂_0`` of shape (B, T, F).

Conditioning:
* Diffusion timestep ``t`` (int per sample) → sinusoidal + MLP embedding.
* Text embedding from a frozen CLIP encoder projected to ``d_model``.

The two are added and prepended as the first token of the sequence. The
transformer attends across the time axis; the output corresponding to the
condition token is discarded.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from wisp.config import ModelConfig
from wisp.data.skeleton import NUM_JOINTS


def _sinusoidal(timesteps: Tensor, dim: int) -> Tensor:
    half = dim // 2
    device = timesteps.device
    exponents = torch.arange(half, device=device, dtype=torch.float32) * (
        -math.log(10000.0) / half
    )
    args = timesteps.float().unsqueeze(-1) * exponents.exp().unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        emb = _sinusoidal(t, self.d_model)
        return self.mlp(emb)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.shape[1]]


class MotionDenoiser(nn.Module):
    """Predicts the clean motion sample ``x̂_0`` (not noise — MDM convention)."""

    def __init__(self, cfg: ModelConfig, text_embed_dim: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_dim = NUM_JOINTS * cfg.rot_repr_dim + cfg.root_dim

        self.in_proj = nn.Linear(self.feature_dim, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, self.feature_dim)
        self.pos_enc = PositionalEncoding(cfg.d_model)
        self.t_embed = TimestepEmbedding(cfg.d_model)

        text_dim = text_embed_dim if text_embed_dim is not None else cfg.d_model
        self.cond_proj = nn.Linear(text_dim, cfg.d_model)
        # Learned "null" condition for classifier-free guidance.
        self.null_cond = nn.Parameter(torch.zeros(cfg.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: Tensor,                # (B, T, F)
        timesteps: Tensor,        # (B,)
        text_emb: Tensor | None,  # (B, text_dim) or None for full drop
        cond_mask: Tensor | None = None,  # (B,) bool — True = use text, False = null
    ) -> Tensor:
        b, t_len, _ = x.shape
        device = x.device

        t_emb = self.t_embed(timesteps.to(device))                 # (B, d_model)
        if text_emb is not None:
            c_emb = self.cond_proj(text_emb.to(device))            # (B, d_model)
        else:
            c_emb = self.null_cond.unsqueeze(0).expand(b, -1)
        if cond_mask is not None:
            mask = cond_mask.to(device).float().unsqueeze(-1)
            c_emb = mask * c_emb + (1.0 - mask) * self.null_cond.unsqueeze(0).expand_as(c_emb)
        cond_token = (t_emb + c_emb).unsqueeze(1)                  # (B, 1, d_model)

        h = self.in_proj(x)                                        # (B, T, d_model)
        h = self.pos_enc(h)
        h = torch.cat([cond_token, h], dim=1)                      # (B, T+1, d_model)
        h = self.transformer(h)
        h = self.out_norm(h[:, 1:])                                # drop cond token
        return self.out_proj(h)
