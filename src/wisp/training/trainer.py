"""Main training loop for the motion diffusion model."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from wisp.config import WispConfig
from wisp.data.dataset import CMUMotionDataset, collate_motion
from wisp.models.denoiser import MotionDenoiser
from wisp.models.diffusion import DiffusionScheduler
from wisp.models.text_encoder import CLIPTextEncoder
from wisp.training.ema import EMA
from wisp.training.losses import diffusion_loss, join_motion
from wisp.utils.logging import get_device, log_environment


def _maybe_drop_text(batch_size: int, prob: float, device: torch.device) -> torch.Tensor:
    """``True`` = keep text conditioning; ``False`` = use null embedding."""
    return torch.rand(batch_size, device=device) > prob


def run_training(
    dataset_dir: Path,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    device_override: str | None,
    num_workers: int | None,
    resume: Path | None,
    use_compile: bool,
) -> None:
    cfg = WispConfig()
    device = torch.device(get_device(device_override))
    log_environment(str(device))

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    dataset = CMUMotionDataset(
        dataset_dir,
        num_frames=cfg.data.num_frames,
        augment_flip=cfg.data.augment_flip,
        augment_yaw=cfg.data.augment_rotation,
        seed=cfg.data.seed,
    )
    logger.info(f"dataset: {len(dataset)} virtual windows across {len(dataset.entries)} clips")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if num_workers is not None else cfg.train.num_workers,
        collate_fn=collate_motion,
        drop_last=True,
    )

    text_encoder = CLIPTextEncoder(cfg.model.clip_arch, cfg.model.clip_pretrained, cfg.model.d_model)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()  # CLIP frozen; only the projection is trained as part of the encoder

    denoiser = MotionDenoiser(cfg.model, text_embed_dim=cfg.model.d_model).to(device)
    ema = EMA(denoiser, decay=cfg.train.ema_decay)
    scheduler = DiffusionScheduler(cfg.diffusion)
    scheduler.to(device)

    optim_params = list(denoiser.parameters()) + list(text_encoder.projection.parameters())
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=cfg.train.lr,
        betas=cfg.train.adam_betas,
        weight_decay=cfg.train.weight_decay,
    )

    start_epoch = 0
    if resume is not None and resume.exists():
        logger.info(f"resuming from {resume}")
        state = torch.load(resume, map_location=device)
        denoiser.load_state_dict(state["denoiser"])
        ema.load_state_dict(state["ema"])
        text_encoder.projection.load_state_dict(state["text_projection"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1

    if use_compile and device.type == "cuda":
        denoiser = torch.compile(denoiser, mode="reduce-overhead")

    logger.info(f"denoiser params: {sum(p.numel() for p in denoiser.parameters()) / 1e6:.2f}M")

    step = 0
    for epoch in range(start_epoch, epochs):
        denoiser.train()
        t0 = time.time()
        epoch_metrics = {"total": 0.0, "rec": 0.0, "pos": 0.0, "vel": 0.0, "accel": 0.0, "bone": 0.0}
        for batch in loader:
            rot6d = batch["rotations_6d"].to(device)
            root = batch["root_translation"].to(device)
            positions = batch["positions"].to(device)
            bone_offsets = batch["bone_offsets"].to(device)
            prompts: list[str] = batch["prompts"]

            x0 = join_motion(rot6d, root)
            t = scheduler.sample_timesteps(x0.shape[0], device)
            noise = torch.randn_like(x0)
            x_t = scheduler.q_sample(x0, t, noise)

            with torch.no_grad():
                text_emb = text_encoder(prompts)
            cond_mask = _maybe_drop_text(x0.shape[0], cfg.model.cond_drop_prob, device)

            x0_pred = denoiser(x_t, t, text_emb, cond_mask=cond_mask)
            losses = diffusion_loss(
                x0_pred,
                x0,
                positions,
                bone_offsets,
                lambda_pos=cfg.train.lambda_pos,
                lambda_vel=cfg.train.lambda_vel,
                lambda_accel=cfg.train.lambda_accel,
                lambda_bone=cfg.train.lambda_bone,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(optim_params, cfg.train.grad_clip)
            optimizer.step()
            ema.update(denoiser)

            for k in epoch_metrics:
                epoch_metrics[k] += losses[k].detach().item()
            step += 1

        n_steps = max(1, len(loader))
        for k in epoch_metrics:
            epoch_metrics[k] /= n_steps
        dt = time.time() - t0
        logger.info(
            f"epoch {epoch + 1:>4}/{epochs} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items())
            + f" | {dt:.1f}s"
        )

        if (epoch + 1) % cfg.train.ckpt_every == 0 or epoch + 1 == epochs:
            state = {
                "epoch": epoch,
                "denoiser": denoiser.state_dict(),
                "ema": ema.state_dict(),
                "text_projection": text_encoder.projection.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(state, ckpt_dir / "last.pt")
            torch.save(
                {"denoiser_ema": ema.state_dict(), "text_projection": text_encoder.projection.state_dict(), "config": cfg},
                ckpt_dir / "ema.pt",
            )
            logger.info(f"checkpoint saved (epoch {epoch + 1})")
