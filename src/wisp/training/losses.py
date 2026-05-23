"""Motion losses: diffusion + position + velocity + bone length."""

from __future__ import annotations

import torch
from torch import Tensor

from wisp.data.kinematics import fk_from_6d
from wisp.data.skeleton import NUM_JOINTS, PARENTS


def split_motion(x: Tensor) -> tuple[Tensor, Tensor]:
    """Split a flat motion tensor into (rotations_6d, root_translation).

    x : (B, T, F) where F = J*6 + 3
    -> rotations_6d (B, T, J, 6), root_translation (B, T, 3)
    """
    b, t, _ = x.shape
    rot = x[..., : NUM_JOINTS * 6].reshape(b, t, NUM_JOINTS, 6)
    root = x[..., NUM_JOINTS * 6 :]
    return rot, root


def join_motion(rotations_6d: Tensor, root_translation: Tensor) -> Tensor:
    b, t, _, _ = rotations_6d.shape
    return torch.cat([rotations_6d.reshape(b, t, NUM_JOINTS * 6), root_translation], dim=-1)


def velocity(positions: Tensor) -> Tensor:
    """Frame-to-frame velocity (forward differences)."""
    return positions[:, 1:] - positions[:, :-1]


def acceleration(positions: Tensor) -> Tensor:
    """Frame-to-frame acceleration (second forward difference)."""
    return positions[:, 2:] - 2.0 * positions[:, 1:-1] + positions[:, :-2]


def bone_lengths(positions: Tensor) -> Tensor:
    """Per-bone Euclidean length, shape (B, T, J-1)."""
    parents = torch.tensor([p for p in PARENTS if p >= 0], dtype=torch.long, device=positions.device)
    child_idx = torch.tensor(
        [j for j, p in enumerate(PARENTS) if p >= 0], dtype=torch.long, device=positions.device
    )
    bones = positions.index_select(2, child_idx) - positions.index_select(2, parents)
    return bones.norm(dim=-1)


def diffusion_loss(
    x0_pred: Tensor,
    x0_true: Tensor,
    positions_true: Tensor,
    bone_offsets: Tensor,
    *,
    lambda_pos: float,
    lambda_vel: float,
    lambda_accel: float,
    lambda_bone: float,
) -> dict[str, Tensor]:
    """Composite training loss.

    Returns a dict with ``total`` plus the individual components for logging.
    """
    rec = (x0_pred - x0_true).pow(2).mean()

    rot_pred, root_pred = split_motion(x0_pred)
    positions_pred = fk_from_6d(rot_pred, root_pred, bone_offsets)

    pos_err = (positions_pred - positions_true).pow(2).mean()
    vel_err = (velocity(positions_pred) - velocity(positions_true)).pow(2).mean()
    accel_err = (acceleration(positions_pred) - acceleration(positions_true)).pow(2).mean()
    bones_pred = bone_lengths(positions_pred)
    bones_true = bone_lengths(positions_true)
    bone_err = (bones_pred - bones_true).pow(2).mean()

    total = (
        rec
        + lambda_pos * pos_err
        + lambda_vel * vel_err
        + lambda_accel * accel_err
        + lambda_bone * bone_err
    )
    return {
        "total": total,
        "rec": rec.detach(),
        "pos": pos_err.detach(),
        "vel": vel_err.detach(),
        "accel": accel_err.detach(),
        "bone": bone_err.detach(),
    }
