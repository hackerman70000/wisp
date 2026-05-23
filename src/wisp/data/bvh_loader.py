"""Load CMU mocap BVH files and extract our 15-joint stickman trajectory."""

from __future__ import annotations

from pathlib import Path

import bvhio
import numpy as np

from wisp.data.skeleton import BVH_JOINT_NAMES, NUM_JOINTS, Joint


def _find_joint(root, name: str):
    matches = root.filter(name)
    if not matches:
        raise KeyError(f"joint '{name}' not found in BVH hierarchy")
    return matches[0]


def load_bvh_positions(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Load a BVH file and return world positions for our 15 joints.

    Returns
    -------
    positions : np.ndarray of shape ``(T, 15, 3)`` in BVH world space (cm).
    fps : float — source frame rate.
    rest_positions : np.ndarray of shape ``(15, 3)`` — joint positions in the
        BVH rest (T) pose. Used to compute anatomically correct bone offsets.
    """
    root = bvhio.readAsHierarchy(str(path))
    bvh_meta = bvhio.readAsBvh(str(path))
    fps = 1.0 / bvh_meta.FrameTime

    joint_handles = [_find_joint(root, BVH_JOINT_NAMES[Joint(i)]) for i in range(NUM_JOINTS)]

    root.loadRestPose(recursive=True)
    rest_positions = np.stack(
        [np.asarray(h.PositionWorld, dtype=np.float32) for h in joint_handles]
    )

    num_frames = len(root.Keyframes)
    positions = np.empty((num_frames, NUM_JOINTS, 3), dtype=np.float32)
    for frame in range(num_frames):
        root.loadPose(frame)
        for j, handle in enumerate(joint_handles):
            positions[frame, j] = np.asarray(handle.PositionWorld, dtype=np.float32)

    return positions, float(fps), rest_positions


def canonicalize_rest(rest_positions: np.ndarray, scale: float) -> np.ndarray:
    """Re-center & rescale a rest pose with the same factor used for sequences."""
    rest = rest_positions.copy()
    rest[..., [0, 2]] -= rest[int(Joint.PELVIS), [0, 2]]
    rest /= scale
    return rest.astype(np.float32)


def canonicalize(positions: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert CMU BVH world positions to a canonical coordinate frame.

    - Re-axes from BVH (Y up, cm) to our convention (Y up, scaled units).
    - Centers each frame so that the PELVIS sits at the origin in X/Z;
      the global Y of PELVIS is preserved so jumps stay vertical.
    - Rescales so that the median rest skeleton height (head-pelvis distance)
      is 1.0 — matches the ``set_xlim(-1.5, 1.5)`` viewport from the spec.
    """
    pelvis_xz = positions[:, Joint.PELVIS, [0, 2]][:, None, :]  # (T, 1, 2)
    positions = positions.copy()
    positions[..., [0, 2]] -= pelvis_xz

    # Scale so the rest-pose body height (head ↦ mean ankle) is 1.0 — keeps
    # the figure inside the ``[-1.5, 1.5]`` viewport from the spec.
    ankles = 0.5 * (positions[:, Joint.RIGHT_ANKLE] + positions[:, Joint.LEFT_ANKLE])
    body_height = float(np.median(np.linalg.norm(positions[:, Joint.HEAD] - ankles, axis=-1)))
    scale = body_height if body_height > 1e-6 else 1.0
    positions /= scale

    # Subtract the median Y of PELVIS so the figure stands centered vertically.
    pelvis_y = float(np.median(positions[:, Joint.PELVIS, 1]))
    positions[..., 1] -= pelvis_y
    return positions.astype(np.float32), scale
