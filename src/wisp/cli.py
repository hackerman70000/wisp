from pathlib import Path
from typing import Annotated

import typer

from wisp.config import DEFAULT_DATASET_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED

app = typer.Typer(
    name="wisp",
    help="Text-conditional stick-figure motion diffusion.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command("download-data")
def download_data() -> None:
    """Download the curated walk / jump BVH subset from the CMU mocap mirror."""
    from wisp.data.catalog import CLIPS_BY_LABEL, bvh_url, clip_name
    from wisp.utils.logging import setup_logging
    import time
    import requests
    from loguru import logger

    setup_logging(DEFAULT_OUTPUT_DIR / "logs", run_name="download")
    raw_dir = DEFAULT_DATASET_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in CLIPS_BY_LABEL.values())
    done = 0
    for label, clips in CLIPS_BY_LABEL.items():
        label_dir = raw_dir / label
        label_dir.mkdir(exist_ok=True)
        for subject, trial in clips:
            dst = label_dir / f"{clip_name(subject, trial)}.bvh"
            done += 1
            if dst.exists() and dst.stat().st_size > 0:
                continue
            for attempt in range(3):
                try:
                    r = requests.get(bvh_url(subject, trial), timeout=30)
                    r.raise_for_status()
                    dst.write_bytes(r.content)
                    break
                except requests.RequestException as exc:
                    logger.warning(f"{dst.name} retry {attempt + 1}/3: {exc}")
                    time.sleep(1.5 ** (attempt + 1))
            logger.info(f"[{done:>3}/{total}] {label}/{dst.name}")


@app.command("prepare-data")
def prepare_data(
    raw: Annotated[Path, typer.Option(help="Raw BVH directory.")] = DEFAULT_DATASET_DIR / "raw",
    out: Annotated[Path, typer.Option(help="Processed output directory.")] = DEFAULT_DATASET_DIR / "processed",
) -> None:
    """Canonicalise + resample BVH clips into rotation tensors."""
    from wisp.data.preprocess import preprocess_all
    from wisp.utils.logging import setup_logging

    setup_logging(DEFAULT_OUTPUT_DIR / "logs", run_name="prepare")
    preprocess_all(raw, out)


@app.command("train")
def train(
    dataset: Annotated[Path, typer.Option(help="Processed dataset directory.")] = DEFAULT_DATASET_DIR / "processed",
    output: Annotated[Path, typer.Option(help="Checkpoints / logs directory.")] = DEFAULT_OUTPUT_DIR,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 400,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 32,
    resume: Annotated[Path | None, typer.Option(help="Resume from checkpoint.")] = None,
    device: Annotated[str | None, typer.Option(help="cuda | mps | cpu (auto).")] = None,
    num_workers: Annotated[int | None, typer.Option(help="DataLoader workers (auto).")] = None,
    compile_: Annotated[bool, typer.Option("--compile/--no-compile", help="torch.compile (CUDA).")] = False,
) -> None:
    """Train the motion diffusion model."""
    from wisp.training.trainer import run_training
    from wisp.utils.logging import setup_logging

    setup_logging(output / "logs", run_name="train")
    run_training(
        dataset_dir=dataset,
        output_dir=output,
        epochs=epochs,
        batch_size=batch_size,
        resume=resume,
        device_override=device,
        num_workers=num_workers,
        use_compile=compile_,
    )


@app.command("sample")
def sample_cmd(
    ckpt: Annotated[Path, typer.Option(help="EMA checkpoint to sample from.")]
        = DEFAULT_OUTPUT_DIR / "checkpoints" / "ema.pt",
    dataset: Annotated[Path, typer.Option(help="Processed dataset directory.")] = DEFAULT_DATASET_DIR / "processed",
    prompt: Annotated[str, typer.Option(help="Text prompt.")] = "a person is walking",
    out: Annotated[Path, typer.Option(help="Output GIF path.")] = Path("sample.gif"),
    cfg_scale: Annotated[float, typer.Option(help="Classifier-free guidance scale.")] = 1.0,
    num_steps: Annotated[int, typer.Option(help="DDIM sampling steps.")] = 200,
    seed: Annotated[int, typer.Option(help="Random seed.")] = DEFAULT_SEED,
    device: Annotated[str | None, typer.Option(help="cuda | mps | cpu (auto).")] = None,
    body: Annotated[str, typer.Option(help="Body rendering: 'smpl' (mesh) or 'skeleton'.")] = "smpl",
    smooth_window: Annotated[int, typer.Option(help="Savitzky-Golay window (0=off, 5-9 typical).")] = 9,
) -> None:
    """Sample a single animation from a text prompt and save it as a GIF."""
    from wisp.eval.sampler import WispSampler
    from wisp.utils.logging import setup_logging
    from wisp.viz.animate import animate_skeleton_3d, animate_smpl_3d
    from wisp.viz.smpl_renderer import SMPL_NPZ_DEFAULT, load_smpl, vertices_for_motion

    setup_logging(run_name="sample")
    sampler = WispSampler(ckpt, dataset, device_override=device)

    if body == "smpl" and SMPL_NPZ_DEFAULT.exists():
        positions, rot6d, root = sampler.sample_with_rotations(
            [prompt], cfg_scale=cfg_scale, num_steps=num_steps, seed=seed,
            smooth_window=smooth_window,
        )
        smpl_model = load_smpl(device=str(sampler.device))
        verts = vertices_for_motion(rot6d[0], root[0], smpl_model).cpu().numpy()
        animate_smpl_3d(verts, smpl_model.faces, out, fps=24, title=prompt)
    else:
        if body == "smpl":
            from loguru import logger
            logger.warning(f"SMPL model missing at {SMPL_NPZ_DEFAULT}; falling back to skeleton.")
        positions = sampler.sample_numpy(
            [prompt], cfg_scale=cfg_scale, num_steps=num_steps, seed=seed,
            smooth_window=smooth_window,
        )
        animate_skeleton_3d(positions[0], out, fps=24, title=prompt)


@app.command("eval")
def evaluate(
    ckpt: Annotated[Path, typer.Option(help="EMA checkpoint.")]
        = DEFAULT_OUTPUT_DIR / "checkpoints" / "ema.pt",
    dataset: Annotated[Path, typer.Option(help="Processed dataset directory.")] = DEFAULT_DATASET_DIR / "processed",
    output: Annotated[Path, typer.Option(help="Metrics output directory.")] = DEFAULT_OUTPUT_DIR / "eval",
    num_samples: Annotated[int, typer.Option(help="Samples per class for FMD/Var.")] = 32,
    cfg_scale: Annotated[float, typer.Option(help="Classifier-free guidance scale.")] = 2.5,
    seed: Annotated[int, typer.Option(help="Random seed.")] = DEFAULT_SEED,
    device: Annotated[str | None, typer.Option(help="cuda | mps | cpu (auto).")] = None,
) -> None:
    """Compute FMD, MPJPE and Var on generated walks + jumps."""
    from wisp.eval.report import evaluate as eval_fn
    from wisp.utils.logging import setup_logging

    setup_logging(output, run_name="eval")
    eval_fn(ckpt, dataset, output, num_samples=num_samples, cfg_scale=cfg_scale, device=device, seed=seed)


@app.command("report")
def report(
    ckpt: Annotated[Path, typer.Option(help="EMA checkpoint.")]
        = DEFAULT_OUTPUT_DIR / "checkpoints" / "ema.pt",
    dataset: Annotated[Path, typer.Option(help="Processed dataset directory.")] = DEFAULT_DATASET_DIR / "processed",
    output: Annotated[Path, typer.Option(help="Report output directory.")] = DEFAULT_OUTPUT_DIR / "report",
    num_samples: Annotated[int, typer.Option(help="Samples per class for metrics.")] = 32,
    grid_size: Annotated[int, typer.Option(help="Animations per class in the grid.")] = 4,
    cfg_scale: Annotated[float, typer.Option(help="Classifier-free guidance scale.")] = 1.0,
    num_steps: Annotated[int, typer.Option(help="DDIM sampling steps.")] = 200,
    smooth_window: Annotated[int, typer.Option(help="Savitzky-Golay window.")] = 9,
    seed: Annotated[int, typer.Option(help="Random seed.")] = DEFAULT_SEED,
    device: Annotated[str | None, typer.Option(help="cuda | mps | cpu (auto).")] = None,
) -> None:
    """Generate the full report (metrics + GIFs)."""
    from wisp.eval.report import build_report
    from wisp.utils.logging import setup_logging

    setup_logging(output, run_name="report")
    build_report(
        ckpt, dataset, output,
        num_samples=num_samples, grid_size=grid_size,
        cfg_scale=cfg_scale, num_steps=num_steps, smooth_window=smooth_window,
        device=device, seed=seed,
    )


if __name__ == "__main__":
    app()
