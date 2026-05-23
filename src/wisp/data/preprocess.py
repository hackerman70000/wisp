"""Convert raw CMU BVH clips into per-frame rotation tensors on disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from wisp.config import NUM_FRAMES, TARGET_FPS
from wisp.data.bvh_loader import canonicalize, canonicalize_rest, load_bvh_positions
from wisp.data.catalog import CLIPS_BY_LABEL, clip_name
from wisp.data.kinematics import (
    matrix_to_6d,
    positions_to_rotations,
    rest_pose_offsets,
)
from wisp.data.skeleton import NUM_JOINTS


def _resample(positions: np.ndarray, src_fps: float, dst_fps: int) -> np.ndarray:
    """Resample a (T, J, 3) trajectory by linear interpolation along time."""
    if abs(src_fps - dst_fps) < 1e-3:
        return positions
    duration = positions.shape[0] / src_fps
    num_dst = max(1, int(round(duration * dst_fps)))
    src_t = np.linspace(0.0, 1.0, positions.shape[0], dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, num_dst, dtype=np.float64)
    j_count = positions.shape[1]
    out = np.empty((num_dst, j_count, 3), dtype=np.float32)
    for j in range(j_count):
        for axis in range(3):
            out[:, j, axis] = np.interp(dst_t, src_t, positions[:, j, axis])
    return out


def _process_clip(bvh_path: Path) -> dict | None:
    """Load + canonicalize + resample one clip; return ``None`` if it's too short."""
    try:
        positions, fps, rest_positions = load_bvh_positions(bvh_path)
    except Exception as exc:
        logger.error(f"{bvh_path.name}: load failed ({exc})")
        return None

    positions, scale = canonicalize(positions)
    rest_positions = canonicalize_rest(rest_positions, scale)

    positions = _resample(positions, fps, TARGET_FPS)
    if positions.shape[0] < NUM_FRAMES:
        logger.warning(f"{bvh_path.name}: {positions.shape[0]} frames < {NUM_FRAMES}, skipped")
        return None

    bone_offsets = rest_pose_offsets(rest_positions)
    root_translation, rotations = positions_to_rotations(positions, bone_offsets)

    import torch

    rot_torch = torch.from_numpy(rotations)
    rotations_6d = matrix_to_6d(rot_torch).numpy().astype(np.float32)

    return {
        "positions": positions.astype(np.float32),
        "rotations_6d": rotations_6d,
        "root_translation": root_translation.astype(np.float32),
        "bone_offsets": bone_offsets.astype(np.float32),
    }


def preprocess_all(raw_dir: Path, processed_dir: Path) -> None:
    """Process every clip in ``raw_dir/<label>/*.bvh`` into ``processed_dir``."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    bone_offsets_accum: list[np.ndarray] = []

    for label, clips in CLIPS_BY_LABEL.items():
        label_dir = processed_dir / label
        label_dir.mkdir(exist_ok=True)
        for subject, trial in clips:
            name = clip_name(subject, trial)
            bvh_path = raw_dir / label / f"{name}.bvh"
            if not bvh_path.exists():
                logger.warning(f"missing raw clip: {bvh_path}")
                continue

            data = _process_clip(bvh_path)
            if data is None:
                continue

            out_path = label_dir / f"{name}.npz"
            np.savez(out_path, **data)
            bone_offsets_accum.append(data["bone_offsets"])
            index.append(
                {
                    "label": label,
                    "subject": subject,
                    "trial": trial,
                    "frames": int(data["positions"].shape[0]),
                    "path": str(out_path.relative_to(processed_dir)),
                }
            )
            logger.info(f"processed {label}/{name} : {data['positions'].shape[0]} frames")

    if bone_offsets_accum:
        mean_offsets = np.stack(bone_offsets_accum).mean(axis=0).astype(np.float32)
        np.save(processed_dir / "mean_bone_offsets.npy", mean_offsets)

    (processed_dir / "index.json").write_text(json.dumps(index, indent=2))
    logger.success(
        f"done — {len(index)} clips ({sum(1 for x in index if x['label'] == 'walk')} walks, "
        f"{sum(1 for x in index if x['label'] == 'jump')} jumps)"
    )


def load_mean_bone_offsets(processed_dir: Path) -> np.ndarray:
    arr = np.load(processed_dir / "mean_bone_offsets.npy")
    if arr.shape != (NUM_JOINTS, 3):
        raise ValueError(f"mean_bone_offsets has wrong shape: {arr.shape}")
    return arr.astype(np.float32)
