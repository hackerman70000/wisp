"""Convenience wrapper: load checkpoint → produce motion tensors from prompts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy.signal import savgol_filter
from torch import Tensor

from wisp.config import NUM_FRAMES, WispConfig
from wisp.data.kinematics import fk_from_6d
from wisp.data.preprocess import load_mean_bone_offsets
from wisp.data.skeleton import NUM_JOINTS
from wisp.models.denoiser import MotionDenoiser
from wisp.models.diffusion import DiffusionScheduler, sample as ddim_sample
from wisp.models.text_encoder import CLIPTextEncoder
from wisp.training.losses import split_motion
from wisp.utils.logging import get_device


class WispSampler:
    """Produces (T, J, 3) position tensors from text prompts."""

    def __init__(
        self,
        ckpt_path: Path,
        processed_dir: Path,
        device_override: str | None = None,
    ) -> None:
        self.device = torch.device(get_device(device_override))
        logger.info(f"loading {ckpt_path} on {self.device}")

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cfg: WispConfig = state.get("config", WispConfig())

        self.text_encoder = CLIPTextEncoder(
            self.cfg.model.clip_arch, self.cfg.model.clip_pretrained, self.cfg.model.d_model
        )
        self.text_encoder.projection.load_state_dict(state["text_projection"])
        self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder.eval()

        self.denoiser = MotionDenoiser(self.cfg.model, text_embed_dim=self.cfg.model.d_model)
        self.denoiser.load_state_dict(state["denoiser_ema"])
        self.denoiser = self.denoiser.to(self.device)
        self.denoiser.eval()

        self.scheduler = DiffusionScheduler(self.cfg.diffusion)
        self.scheduler.to(self.device)
        self.bone_offsets = torch.from_numpy(load_mean_bone_offsets(processed_dir)).to(self.device)

    @torch.no_grad()
    def sample(
        self,
        prompts: list[str],
        cfg_scale: float | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        smooth_window: int = 0,
        smooth_poly: int = 2,
    ) -> Tensor:
        """Return positions of shape (B, T, J, 3) for ``len(prompts)`` motions.

        ``smooth_window``: Savitzky-Golay window length applied to the 6D
        rotations and root translation along time before FK. ``0`` disables
        smoothing. Must be odd and ``≤`` ``NUM_FRAMES``. Typical: 5-9 frames.
        """
        if seed is not None:
            torch.manual_seed(seed)

        b = len(prompts)
        text_emb = self.text_encoder(prompts)
        null_emb = self.text_encoder([""] * b)

        feature_dim = NUM_JOINTS * 6 + 3
        shape = (b, NUM_FRAMES, feature_dim)
        x0 = ddim_sample(
            self.denoiser,
            self.scheduler,
            shape,
            text_emb,
            device=self.device,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            null_text_emb=null_emb,
        )
        rot, root = split_motion(x0)
        if smooth_window and smooth_window > 1:
            rot, root = _smooth_motion(rot, root, smooth_window, smooth_poly)
        positions = fk_from_6d(rot, root, self.bone_offsets)
        return positions

    def sample_numpy(self, *args, **kwargs) -> np.ndarray:
        return self.sample(*args, **kwargs).cpu().numpy()

    @torch.no_grad()
    def sample_with_rotations(
        self,
        prompts: list[str],
        cfg_scale: float | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        smooth_window: int = 0,
        smooth_poly: int = 2,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Like :meth:`sample` but also returns 6D rotations and root translation."""
        if seed is not None:
            torch.manual_seed(seed)
        b = len(prompts)
        text_emb = self.text_encoder(prompts)
        null_emb = self.text_encoder([""] * b)
        feature_dim = NUM_JOINTS * 6 + 3
        shape = (b, NUM_FRAMES, feature_dim)
        x0 = ddim_sample(
            self.denoiser, self.scheduler, shape, text_emb,
            device=self.device, cfg_scale=cfg_scale, num_steps=num_steps,
            null_text_emb=null_emb,
        )
        rot, root = split_motion(x0)
        if smooth_window and smooth_window > 1:
            rot, root = _smooth_motion(rot, root, smooth_window, smooth_poly)
        positions = fk_from_6d(rot, root, self.bone_offsets)
        return positions, rot, root


def _smooth_motion(
    rot_6d: Tensor, root: Tensor, window: int, poly: int
) -> tuple[Tensor, Tensor]:
    """Savitzky-Golay smoothing along the time axis for rotations + root.

    Smoothing the **6D rotations** (not the rotation matrices) is the cleanest
    place: Gram-Schmidt downstream still produces orthonormal matrices, so
    bone lengths are preserved. Smoothing positions directly would slightly
    stretch the bones.
    """
    if window % 2 == 0:
        window += 1
    t = rot_6d.shape[1]
    window = min(window, t - 1 if t % 2 == 0 else t)
    if window <= poly:
        return rot_6d, root

    rot_np = rot_6d.detach().cpu().numpy()                                   # (B, T, J, 6)
    root_np = root.detach().cpu().numpy()                                    # (B, T, 3)
    rot_np = savgol_filter(rot_np, window_length=window, polyorder=poly, axis=1)
    root_np = savgol_filter(root_np, window_length=window, polyorder=poly, axis=1)
    rot_out = torch.from_numpy(rot_np).to(device=rot_6d.device, dtype=rot_6d.dtype)
    root_out = torch.from_numpy(root_np).to(device=root.device, dtype=root.dtype)
    return rot_out, root_out
