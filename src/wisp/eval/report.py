"""Build report artefacts: metric table + visual samples + GT reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from wisp.data.dataset import LABEL_PROMPTS
from wisp.eval.metrics import fmd, mpjpe_to_real, sample_variance
from wisp.eval.sampler import WispSampler
from wisp.viz.animate import animate_grid, animate_skeleton_3d, animate_smpl_3d
from wisp.viz.smpl_renderer import SMPL_NPZ_DEFAULT, load_smpl, vertices_for_motion


def _most_active_window(positions: np.ndarray, num_frames: int) -> np.ndarray:
    """Pick the ``num_frames``-frame window with the highest kinetic energy.

    For walks this just lands on any cycle; for jumps it reliably finds the
    take-off + flight + landing (the most energetic ~2s of the clip).
    """
    if positions.shape[0] == num_frames:
        return positions
    velocity_sq = np.diff(positions, axis=0).reshape(positions.shape[0] - 1, -1)
    energy = (velocity_sq ** 2).sum(axis=-1)  # (T-1,)
    kernel = np.ones(num_frames - 1, dtype=np.float64)
    sliding = np.convolve(energy, kernel, mode="valid")  # (T - num_frames + 1,)
    start = int(np.argmax(sliding))
    return positions[start : start + num_frames]


def _anchor_xz(positions: np.ndarray) -> np.ndarray:
    """Subtract the first-frame pelvis XZ — same convention as the dataset."""
    from wisp.data.skeleton import Joint
    out = positions.copy()
    out[..., 0] -= positions[0, int(Joint.PELVIS), 0]
    out[..., 2] -= positions[0, int(Joint.PELVIS), 2]
    return out


def _load_real_motions(processed_dir: Path, label: str) -> list[np.ndarray]:
    """Return one canonical 48-frame window per real clip (most active part)."""
    label_dir = processed_dir / label
    motions = []
    for npz_path in sorted(label_dir.glob("*.npz")):
        with np.load(npz_path) as data:
            positions = data["positions"]
        if positions.shape[0] < 48:
            continue
        window = _most_active_window(positions.astype(np.float32), 48)
        motions.append(_anchor_xz(window))
    return motions


def _compute_metrics(
    sampler: WispSampler,
    processed_dir: Path,
    *,
    num_samples: int,
    cfg_scale: float,
    num_steps: int,
    smooth_window: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for label, prompts in LABEL_PROMPTS.items():
        real = _load_real_motions(processed_dir, label)
        if not real:
            logger.warning(f"no real {label} clips found; skipping")
            continue
        prompt = prompts[0]
        torch.manual_seed(seed)
        generated = sampler.sample_numpy(
            [prompt] * num_samples,
            cfg_scale=cfg_scale, num_steps=num_steps, smooth_window=smooth_window,
        )
        gen_list = [generated[i] for i in range(num_samples)]
        results[label] = {
            "FMD": fmd(real, gen_list),
            "MPJPE": mpjpe_to_real(gen_list, real),
            "Var": sample_variance(gen_list),
        }
        logger.info(
            f"{label}: FMD={results[label]['FMD']:.4f}  "
            f"MPJPE={results[label]['MPJPE']:.4f}  Var={results[label]['Var']:.4f}"
        )
    return results


def evaluate(
    ckpt: Path,
    processed_dir: Path,
    output_dir: Path,
    *,
    num_samples: int = 32,
    cfg_scale: float = 1.0,
    num_steps: int = 200,
    smooth_window: int = 9,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = WispSampler(ckpt, processed_dir, device_override=device)
    results = _compute_metrics(
        sampler, processed_dir,
        num_samples=num_samples, cfg_scale=cfg_scale,
        num_steps=num_steps, smooth_window=smooth_window, seed=seed,
    )
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def build_report(
    ckpt: Path,
    processed_dir: Path,
    output_dir: Path,
    *,
    num_samples: int = 32,
    grid_size: int = 4,
    cfg_scale: float = 1.0,
    num_steps: int = 200,
    smooth_window: int = 9,
    device: str | None = None,
    seed: int = 0,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = WispSampler(ckpt, processed_dir, device_override=device)

    metrics = _compute_metrics(
        sampler, processed_dir,
        num_samples=num_samples, cfg_scale=cfg_scale,
        num_steps=num_steps, smooth_window=smooth_window, seed=seed,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    smpl_model = None
    if SMPL_NPZ_DEFAULT.exists():
        smpl_model = load_smpl(device=str(sampler.device))
        logger.info("SMPL model found — single-sample GIFs will use the body mesh")
    else:
        logger.warning(
            f"{SMPL_NPZ_DEFAULT} missing; falling back to line+dot skeleton. "
            f"Run `uv run python scripts/setup_smpl.py` after placing the .pkl."
        )

    for label, prompts in LABEL_PROMPTS.items():
        prompt = prompts[0]
        torch.manual_seed(seed)
        positions = sampler.sample_numpy(
            [prompt] * grid_size,
            cfg_scale=cfg_scale, num_steps=num_steps, smooth_window=smooth_window,
        )
        gen_list = [positions[i] for i in range(grid_size)]
        animate_grid(
            gen_list,
            output_dir / f"grid_{label}.gif",
            titles=[f"{prompt} (sample {i + 1})" for i in range(grid_size)],
            fps=24,
            cols=grid_size,
        )

        if smpl_model is not None:
            torch.manual_seed(seed)
            _, rot6d, root = sampler.sample_with_rotations(
                [prompt],
                cfg_scale=cfg_scale, num_steps=num_steps, smooth_window=smooth_window,
            )
            verts = vertices_for_motion(rot6d[0], root[0], smpl_model).cpu().numpy()
            animate_smpl_3d(
                verts, smpl_model.faces,
                output_dir / f"sample_{label}.gif",
                fps=24, title=prompt,
            )
        else:
            animate_skeleton_3d(
                gen_list[0],
                output_dir / f"sample_{label}.gif",
                fps=24, title=prompt,
            )

        real = _load_real_motions(processed_dir, label)
        if real:
            animate_skeleton_3d(
                real[0],
                output_dir / f"gt_{label}.gif",
                fps=24,
                title=f"GT {label}",
            )

    _write_table(metrics, output_dir / "metrics.md")
    logger.success(f"report saved under {output_dir}")


def _write_table(metrics: dict[str, dict[str, float]], path: Path) -> None:
    lines = [
        "| Motion | FMD | MPJPE | Var |",
        "|--------|----:|------:|----:|",
    ]
    for label, vals in metrics.items():
        lines.append(f"| {label} | {vals['FMD']:.4f} | {vals['MPJPE']:.4f} | {vals['Var']:.4f} |")
    path.write_text("\n".join(lines) + "\n")
