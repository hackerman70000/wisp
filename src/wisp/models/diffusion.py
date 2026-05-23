"""DDPM scheduler with cosine schedule, x0-prediction and CFG sampling.

The denoiser predicts the clean sample ``x_0`` directly (MDM convention).
Training noises the clean motion to a random ``t``; loss is computed between
the predicted and the original clean motion. Sampling iterates from
``x_T ~ N(0, I)`` toward ``x_0`` using the DDIM update with optional
classifier-free guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from wisp.config import DiffusionConfig


def _cosine_betas(num_steps: int, s: float = 0.008) -> Tensor:
    """Cosine β schedule from [Nichol & Dhariwal 2021]."""
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999).float()


@dataclass
class DiffusionTensors:
    betas: Tensor                # (T,)
    alphas: Tensor               # (T,)
    alphas_cumprod: Tensor       # (T,)
    sqrt_alphas_cumprod: Tensor  # (T,)
    sqrt_one_minus_alphas_cumprod: Tensor  # (T,)


def make_schedule(cfg: DiffusionConfig) -> DiffusionTensors:
    if cfg.beta_schedule == "cosine":
        betas = _cosine_betas(cfg.num_train_timesteps)
    elif cfg.beta_schedule == "linear":
        betas = torch.linspace(1e-4, 0.02, cfg.num_train_timesteps)
    else:
        raise ValueError(f"unknown beta schedule: {cfg.beta_schedule}")
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionTensors(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=alphas_cumprod.sqrt(),
        sqrt_one_minus_alphas_cumprod=(1.0 - alphas_cumprod).sqrt(),
    )


def _extract(arr: Tensor, t: Tensor, shape: tuple[int, ...]) -> Tensor:
    out = arr.to(t.device)[t]
    return out.reshape(-1, *([1] * (len(shape) - 1)))


class DiffusionScheduler:
    """Pure-math diffusion: forward q(x_t | x_0), DDIM reverse step."""

    def __init__(self, cfg: DiffusionConfig) -> None:
        self.cfg = cfg
        self.tensors = make_schedule(cfg)
        self.num_train_timesteps = cfg.num_train_timesteps

    def to(self, device: torch.device) -> None:
        self.tensors = DiffusionTensors(
            **{k: getattr(self.tensors, k).to(device) for k in self.tensors.__dataclass_fields__}
        )

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: ``x_t = sqrt(α̅_t) x_0 + sqrt(1-α̅_t) ε``."""
        sa = _extract(self.tensors.sqrt_alphas_cumprod, t, x0.shape)
        som = _extract(self.tensors.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sa * x0 + som * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    # ----------------------- DDIM sampling --------------------------------

    def ddim_step(
        self, x_t: Tensor, x0_pred: Tensor, t: int, t_prev: int | None
    ) -> Tensor:
        device = x_t.device
        alpha_bar_t = self.tensors.alphas_cumprod[t].to(device)
        if t_prev is None:
            return x0_pred  # final step
        alpha_bar_prev = self.tensors.alphas_cumprod[t_prev].to(device)
        eps = (x_t - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt()
        return alpha_bar_prev.sqrt() * x0_pred + (1.0 - alpha_bar_prev).sqrt() * eps

    def ddim_schedule(self, num_sample_steps: int) -> list[int]:
        steps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_sample_steps, dtype=torch.long
        ).tolist()
        return steps


@torch.no_grad()
def sample(
    denoiser,
    scheduler: DiffusionScheduler,
    shape: tuple[int, ...],
    text_emb: Tensor | None,
    *,
    device: torch.device,
    cfg_scale: float | None = None,
    num_steps: int | None = None,
    null_text_emb: Tensor | None = None,
) -> Tensor:
    """Iteratively denoise from N(0, I) to a clean motion sample.

    Classifier-free guidance: when ``cfg_scale > 0`` and ``text_emb`` is given,
    the prediction is interpolated as
    ``x̂ = (1 + w) · x̂_cond - w · x̂_uncond``.
    """
    cfg = scheduler.cfg
    cfg_scale = cfg.cfg_scale if cfg_scale is None else cfg_scale
    steps = scheduler.ddim_schedule(num_steps or cfg.num_sample_steps)

    x_t = torch.randn(shape, device=device)
    batch = shape[0]
    for i, t in enumerate(steps):
        t_prev = steps[i + 1] if i + 1 < len(steps) else None
        t_tensor = torch.full((batch,), t, dtype=torch.long, device=device)

        if cfg_scale > 0.0 and text_emb is not None:
            x0_cond = denoiser(x_t, t_tensor, text_emb)
            x0_uncond = denoiser(x_t, t_tensor, null_text_emb)
            x0_pred = (1.0 + cfg_scale) * x0_cond - cfg_scale * x0_uncond
        else:
            x0_pred = denoiser(x_t, t_tensor, text_emb)

        x_t = scheduler.ddim_step(x_t, x0_pred, t, t_prev)
    return x_t
