"""Forward kinematics + 6D rotation utilities.

Coordinate convention: right-handed, ``+Y`` up. A rest-pose (T-pose) is
parameterised by per-joint bone offsets in the parent's local frame; FK is a
hierarchical chain of these offsets through accumulated rotations.

The "6D" rotation is the first two columns of a rotation matrix, recovered
via Gram-Schmidt — continuous on SO(3) and friendly for regression.
[Zhou et al. 2019, "On the Continuity of Rotation Representations"]
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from wisp.data.skeleton import NUM_JOINTS, PARENTS, Joint

EPS = 1e-7

def topo_order() -> list[int]:
    """Return joint indices in topological order (root → leaves)."""
    order = [int(Joint.PELVIS)]
    pending = set(range(NUM_JOINTS)) - {int(Joint.PELVIS)}
    while pending:
        for j in list(pending):
            if PARENTS[j] in order:
                order.append(j)
                pending.discard(j)
                break
    return order


TOPO_ORDER: tuple[int, ...] = tuple(topo_order())


# ----------------------- 6D ↔ 3x3 rotation ------------------------------------

def matrix_to_6d(R: Tensor) -> Tensor:
    """Take the first two columns of a rotation matrix as a 6D vector.

    Input  : (..., 3, 3)
    Output : (..., 6) — concatenation of column 0 and column 1.
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def sixd_to_matrix(x: Tensor) -> Tensor:
    """Recover a rotation matrix from a 6D vector via Gram-Schmidt.

    Input  : (..., 6)
    Output : (..., 3, 3)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + EPS)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


# ----------------------- Forward kinematics -----------------------------------

def forward_kinematics(
    rotations: Tensor,
    root_translation: Tensor,
    bone_offsets: Tensor,
) -> Tensor:
    """Compose joint rotations + root translation into world positions.

    Parameters
    ----------
    rotations : (B, T, J, 3, 3) — local rotation of each joint in its parent's
        frame; ``rotations[..., PELVIS, :, :]`` is the global body rotation.
    root_translation : (B, T, 3) — world position of PELVIS.
    bone_offsets : (J, 3) — shared offsets, or (B, J, 3) for per-sample offsets.

    Returns
    -------
    positions : (B, T, J, 3)
    """
    if rotations.dim() != 5:
        raise ValueError(f"expected rotations of shape (B,T,J,3,3); got {tuple(rotations.shape)}")
    b, t, j_count, _, _ = rotations.shape
    device, dtype = rotations.device, rotations.dtype
    offsets = bone_offsets.to(device=device, dtype=dtype)
    if offsets.dim() == 2:
        offsets = offsets.unsqueeze(0).expand(b, j_count, 3)
    elif offsets.dim() != 3:
        raise ValueError(f"bone_offsets must be (J,3) or (B,J,3); got {tuple(offsets.shape)}")

    world_rots: list[Tensor | None] = [None] * j_count
    positions: list[Tensor | None] = [None] * j_count

    for j in TOPO_ORDER:
        parent = PARENTS[j]
        if parent < 0:
            world_rots[j] = rotations[:, :, j]
            positions[j] = root_translation
        else:
            world_rots[j] = world_rots[parent] @ rotations[:, :, j]
            offset_local = offsets[:, j].unsqueeze(1).unsqueeze(-1).expand(b, t, 3, 1)
            offset_world = (world_rots[parent] @ offset_local).squeeze(-1)
            positions[j] = positions[parent] + offset_world

    return torch.stack(positions, dim=2)


# ------------- Inverse: derive rotations from absolute joint positions ---------

def _align_vectors(rest: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Minimum-twist rotation taking unit vector ``rest`` to ``current``."""
    rest = rest / (np.linalg.norm(rest) + EPS)
    current = current / (np.linalg.norm(current) + EPS)
    dot = float(np.clip(np.dot(rest, current), -1.0, 1.0))
    if dot > 1.0 - EPS:
        return np.eye(3, dtype=np.float32)
    if dot < -1.0 + EPS:
        # 180° flip: pick any axis perpendicular to ``rest``
        perp_seed = np.array([1.0, 0.0, 0.0]) if abs(rest[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(rest, perp_seed)
        axis = axis / (np.linalg.norm(axis) + EPS)
        K = _skew(axis)
        return (np.eye(3) + 2.0 * K @ K).astype(np.float32)
    axis = np.cross(rest, current)
    K = _skew(axis)
    R = np.eye(3) + K + K @ K * (1.0 / (1.0 + dot))
    return R.astype(np.float32)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=np.float32,
    )


def _procrustes_rotation(rest: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Closest rotation (Procrustes) such that ``R @ rest_i ≈ current_i``.

    rest, current : (N, 3) — N pairs of corresponding direction vectors.
    """
    # H = sum_i current_i @ rest_i^T -> R = U V^T from SVD (with reflection fix)
    H = current.T @ rest  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3, dtype=np.float32)
    D[2, 2] = float(np.sign(np.linalg.det(U @ Vt)))
    return (U @ D @ Vt).astype(np.float32)


_CHILDREN_OF: tuple[tuple[int, ...], ...] = tuple(
    tuple(c for c in range(NUM_JOINTS) if PARENTS[c] == j) for j in range(NUM_JOINTS)
)


def positions_to_rotations(
    positions: np.ndarray, bone_offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert world positions to (root_translation, local_rotations).

    A joint's rotation governs the position of its **children** (not its own
    position — that comes from the parent's accumulated rotation). For each
    joint ``j`` we look at the world-frame bone direction to every child
    ``c`` and solve for ``R_j`` such that
    ``world_rot[parent(j)] @ R_j @ offset[c] ≈ pos[c] - pos[j]``.

    Single child: minimum-twist Rodrigues alignment.
    Multiple children: Procrustes (closest rotation to all child directions).
    Leaf: identity (no constraint).

    Parameters
    ----------
    positions : (T, J, 3)
    bone_offsets : (J, 3) — rest-pose bone vectors in parent's local frame.

    Returns
    -------
    root_translation : (T, 3)
    local_rotations : (T, J, 3, 3)
    """
    num_frames = positions.shape[0]
    rotations = np.tile(np.eye(3, dtype=np.float32), (num_frames, NUM_JOINTS, 1, 1))
    world_rot = np.tile(np.eye(3, dtype=np.float32), (num_frames, NUM_JOINTS, 1, 1))
    root_translation = positions[:, int(Joint.PELVIS)].astype(np.float32)

    for t in range(num_frames):
        for j in TOPO_ORDER:
            children = _CHILDREN_OF[j]
            if not children:
                # Leaf — rotation has no observable effect; keep identity.
                if PARENTS[j] >= 0:
                    world_rot[t, j] = world_rot[t, PARENTS[j]]
                continue

            parent = PARENTS[j]
            parent_world = world_rot[t, parent] if parent >= 0 else np.eye(3, dtype=np.float32)

            rest_dirs = np.stack([bone_offsets[c] for c in children]).astype(np.float32)
            world_dirs = np.stack([positions[t, c] - positions[t, j] for c in children]).astype(
                np.float32
            )
            local_dirs = world_dirs @ parent_world  # equivalent to parent_world.T @ world_dirs.T

            if len(children) == 1:
                R_j = _align_vectors(rest_dirs[0], local_dirs[0])
            else:
                R_j = _procrustes_rotation(rest_dirs, local_dirs)

            rotations[t, j] = R_j
            world_rot[t, j] = parent_world @ R_j

    return root_translation, rotations


# ----------------------- Bone offsets from data --------------------------------

def rest_pose_offsets(rest_positions: np.ndarray) -> np.ndarray:
    """Bone offsets in parent's local frame, read straight from the BVH rest pose.

    Because the BVH rest pose is the canonical T-pose with no joint rotations,
    every joint's parent-local frame is just the world frame; ``offset[j]``
    is simply the rest-pose displacement from parent to ``j``.
    """
    offsets = np.zeros_like(rest_positions, dtype=np.float32)
    for j in range(NUM_JOINTS):
        if PARENTS[j] < 0:
            continue
        offsets[j] = rest_positions[j] - rest_positions[PARENTS[j]]
    return offsets


def average_bone_offsets(rest_offsets_per_clip: list[np.ndarray]) -> np.ndarray:
    """Mean rest-pose offsets across multiple clips (subjects)."""
    stack = np.stack(rest_offsets_per_clip)
    return stack.mean(axis=0).astype(np.float32)


# ----------------------- Convenient torch wrappers ----------------------------

def fk_from_6d(
    rotations_6d: Tensor, root_translation: Tensor, bone_offsets: Tensor
) -> Tensor:
    """FK directly from 6D rotations.

    rotations_6d     : (B, T, J, 6)
    root_translation : (B, T, 3)
    bone_offsets     : (J, 3)
    """
    R = sixd_to_matrix(rotations_6d)
    return forward_kinematics(R, root_translation, bone_offsets)
