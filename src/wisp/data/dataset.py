"""PyTorch Dataset wrapping the preprocessed CMU walk/jump clips."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wisp.config import NUM_FRAMES
from wisp.data.kinematics import matrix_to_6d, sixd_to_matrix
from wisp.data.skeleton import LR_SWAP, NUM_JOINTS

# Canonical text prompts used during training. The model conditions on the
# CLIP embedding of one of these for each sample, picked at random.
WALK_PROMPTS: tuple[str, ...] = (
    "a person is walking",
    "the character walks forward",
    "walking motion",
    "a person walking naturally",
    "human walking",
)
JUMP_PROMPTS: tuple[str, ...] = (
    "a person is jumping",
    "the character jumps forward",
    "jumping motion",
    "a person performing a jump",
    "human jump",
)
LABEL_PROMPTS: dict[str, tuple[str, ...]] = {"walk": WALK_PROMPTS, "jump": JUMP_PROMPTS}


@dataclass
class MotionSample:
    rotations_6d: torch.Tensor   # (T, J, 6)
    root_translation: torch.Tensor  # (T, 3)
    positions: torch.Tensor      # (T, J, 3)
    bone_offsets: torch.Tensor   # (J, 3)
    prompt: str
    label: str


def _mirror_rotations(rot_mat: torch.Tensor) -> torch.Tensor:
    """Reflect a sequence of local rotations across the sagittal plane (X→-X)."""
    flipped = rot_mat.clone()
    # Negate the X axis: post-multiply each row by diag(-1,1,1) and pre-multiply.
    flip = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=rot_mat.dtype))
    flipped = flip @ flipped @ flip
    return flipped


def _yaw_rotation(angle: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=dtype,
    )


class CMUMotionDataset(Dataset):
    """Random 48-frame windows from the preprocessed walk / jump clips.

    Each ``__getitem__`` draws a random clip and a random start frame.
    Augmentations:
      * Random left/right mirroring (50% probability)
      * Random global yaw rotation (uniform in ``±yaw_range`` radians)
    """

    def __init__(
        self,
        processed_dir: Path,
        num_frames: int = NUM_FRAMES,
        augment_flip: bool = True,
        augment_yaw: bool = True,
        yaw_range: float = float(np.pi),
        labels: tuple[str, ...] = ("walk", "jump"),
        seed: int = 0,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.num_frames = num_frames
        self.augment_flip = augment_flip
        self.augment_yaw = augment_yaw
        self.yaw_range = yaw_range
        self._rng = random.Random(seed)

        index = json.loads((self.processed_dir / "index.json").read_text())
        self.entries: list[dict] = [e for e in index if e["label"] in labels]
        if not self.entries:
            raise FileNotFoundError(f"no clips with labels={labels} under {processed_dir}")

        # Cache lightweight in-memory copies — the dataset is small enough.
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        for entry in self.entries:
            path = self.processed_dir / entry["path"]
            with np.load(path) as data:
                self._cache[entry["path"]] = {k: data[k].astype(np.float32) for k in data.files}

    def __len__(self) -> int:
        # Virtual length: every clip can supply many overlapping windows;
        # we expose a generous multiplier so an epoch sees enough variety.
        windows_per_clip = max(1, sum(
            max(1, e["frames"] - self.num_frames + 1) // 8 for e in self.entries
        ) // len(self.entries))
        return windows_per_clip * len(self.entries)

    def __getitem__(self, idx: int) -> MotionSample:
        entry = self.entries[idx % len(self.entries)]
        clip = self._cache[entry["path"]]
        total = clip["positions"].shape[0]
        start = self._rng.randint(0, total - self.num_frames)
        end = start + self.num_frames

        positions = torch.from_numpy(clip["positions"][start:end].copy())
        root_translation = torch.from_numpy(clip["root_translation"][start:end].copy())
        rotations_6d = torch.from_numpy(clip["rotations_6d"][start:end].copy())
        bone_offsets = torch.from_numpy(clip["bone_offsets"].copy())

        # Anchor root XZ at the start of the window so absolute world position
        # is not in the conditioning signal.
        root_xz0 = root_translation[0, [0, 2]].clone()
        root_translation[:, 0] -= root_xz0[0]
        root_translation[:, 2] -= root_xz0[1]
        positions[..., 0] -= root_xz0[0]
        positions[..., 2] -= root_xz0[1]

        if self.augment_flip and self._rng.random() < 0.5:
            positions, root_translation, rotations_6d, bone_offsets = self._flip_lr(
                positions, root_translation, rotations_6d, bone_offsets
            )

        if self.augment_yaw:
            angle = self._rng.uniform(-self.yaw_range, self.yaw_range)
            positions, root_translation, rotations_6d = self._apply_yaw(
                angle, positions, root_translation, rotations_6d
            )

        label = entry["label"]
        prompt = self._rng.choice(LABEL_PROMPTS[label])
        return MotionSample(
            rotations_6d=rotations_6d,
            root_translation=root_translation,
            positions=positions,
            bone_offsets=bone_offsets,
            prompt=prompt,
            label=label,
        )

    # ----------------------- augmentations ---------------------------------

    @staticmethod
    def _flip_lr(
        positions: torch.Tensor,
        root_translation: torch.Tensor,
        rotations_6d: torch.Tensor,
        bone_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Mirror everything across the sagittal (YZ) plane.

        - Negate ``x`` for positions, root translation, bone offsets.
        - Swap left/right joints.
        - Reflect each local rotation (R → diag(-1,1,1) R diag(-1,1,1)).
        """
        positions = positions.clone()
        positions[..., 0] *= -1.0
        root_translation = root_translation.clone()
        root_translation[..., 0] *= -1.0
        bone_offsets = bone_offsets.clone()
        bone_offsets[..., 0] *= -1.0

        rot_mat = sixd_to_matrix(rotations_6d)
        rot_mat = _mirror_rotations(rot_mat)
        rotations_6d = matrix_to_6d(rot_mat)

        for a, b in LR_SWAP:
            positions[:, [a, b]] = positions[:, [b, a]]
            rotations_6d[:, [a, b]] = rotations_6d[:, [b, a]]
            bone_offsets[[a, b]] = bone_offsets[[b, a]]
        return positions, root_translation, rotations_6d, bone_offsets

    @staticmethod
    def _apply_yaw(
        angle: float,
        positions: torch.Tensor,
        root_translation: torch.Tensor,
        rotations_6d: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Rotate the whole motion around the world Y axis."""
        R = _yaw_rotation(angle, positions.dtype)
        positions = positions @ R.T  # (T, J, 3) @ (3, 3)
        root_translation = root_translation @ R.T
        rot_mat = sixd_to_matrix(rotations_6d)  # (T, J, 3, 3)
        rot_mat = rot_mat.clone()
        # Pelvis (root) rotation absorbs the yaw — pre-multiply only the root.
        from wisp.data.skeleton import Joint as _Joint
        rot_mat[:, int(_Joint.PELVIS)] = R @ rot_mat[:, int(_Joint.PELVIS)]
        rotations_6d = matrix_to_6d(rot_mat)
        return positions, root_translation, rotations_6d


def collate_motion(samples: list[MotionSample]) -> dict[str, object]:
    return {
        "rotations_6d": torch.stack([s.rotations_6d for s in samples]),
        "root_translation": torch.stack([s.root_translation for s in samples]),
        "positions": torch.stack([s.positions for s in samples]),
        "bone_offsets": torch.stack([s.bone_offsets for s in samples]),
        "prompts": [s.prompt for s in samples],
        "labels": [s.label for s in samples],
    }
