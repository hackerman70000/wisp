"""Evaluation metrics for generated motion.

Three numbers per motion class are reported (see project brief):

* **FMD** — Fréchet distance between low-dim kinematic-feature distributions
  of real vs. generated motions. Lower is better.
* **MPJPE** — mean per-joint position error of each generated sample against
  the closest real motion in the same class. Lower means "more realistic".
* **Var** — average per-joint, per-frame variance across generated samples
  conditioned on the same prompt. Higher means "more creative".
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def _motion_features(positions: np.ndarray) -> np.ndarray:
    """Extract a compact kinematic-feature vector from a motion clip.

    positions : (T, J, 3)
    -> features (F,) — joint-wise velocity / acceleration / vertical-range
       statistics. F = 6 * J.
    """
    if positions.ndim != 3:
        raise ValueError(f"expected (T, J, 3); got {positions.shape}")
    velocity = np.diff(positions, axis=0)            # (T-1, J, 3)
    accel = np.diff(velocity, axis=0)                # (T-2, J, 3)
    speed = np.linalg.norm(velocity, axis=-1)        # (T-1, J)
    accel_mag = np.linalg.norm(accel, axis=-1)       # (T-2, J)

    feats = np.stack(
        [
            speed.mean(axis=0),
            speed.std(axis=0),
            accel_mag.mean(axis=0),
            accel_mag.std(axis=0),
            positions[..., 1].max(axis=0) - positions[..., 1].min(axis=0),  # vertical span
            np.linalg.norm(positions[-1] - positions[0], axis=-1),          # net displacement
        ],
        axis=0,
    )  # (6, J)
    return feats.reshape(-1).astype(np.float64)


def feature_stats(motions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the per-motion feature vectors."""
    feats = np.stack([_motion_features(m) for m in motions])
    mu = feats.mean(axis=0)
    if len(motions) < 2:
        cov = np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
    else:
        cov = np.cov(feats, rowvar=False)
    return mu, cov


def fmd(real: list[np.ndarray], generated: list[np.ndarray]) -> float:
    """Fréchet distance between two distributions of motion features."""
    mu_r, cov_r = feature_stats(real)
    mu_g, cov_g = feature_stats(generated)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))


def mpjpe_to_real(generated: list[np.ndarray], real: list[np.ndarray]) -> float:
    """Mean per-joint position error of each generated sample to the closest
    real motion (within the same class).

    Returns the average across generated samples.
    """
    real_arr = np.stack(real)                                    # (R, T, J, 3)
    distances = []
    for gen in generated:
        diff = np.linalg.norm(real_arr - gen[None], axis=-1)     # (R, T, J)
        per_real = diff.mean(axis=(1, 2))                        # (R,)
        distances.append(per_real.min())
    return float(np.mean(distances))


def sample_variance(samples: list[np.ndarray]) -> float:
    """Per-joint, per-frame variance averaged across all dimensions.

    Operates on multiple samples generated from the same prompt.
    """
    arr = np.stack(samples)                                      # (N, T, J, 3)
    return float(arr.var(axis=0).mean())
