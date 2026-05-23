"""Drape the SMPL body mesh on top of our 15-joint motion.

We do not depend on the ``smplx`` package — it pulls in chumpy, which has a
broken sdist. Instead this module loads the chumpy-free ``.npz`` produced by
``scripts/setup_smpl.py`` and implements a minimal SMPL forward pass (shape
+ pose blend shapes + linear-blend skinning).

Joint mapping: SMPL has 24 joints, we have 15. We feed our rotations into the
matching SMPL joints (pelvis, hips, knees, ankles, neck, head, shoulders,
elbows, wrists) and leave the spine, collar and finger joints at identity.
The result is a body that follows our skeleton — close enough for
visualization, not biomechanically perfect because our T-pose differs from
SMPL's by small offsets at the collar and lower spine.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from wisp.data.skeleton import NUM_JOINTS, Joint

# --- SMPL joint indices (24) ------------------------------------------------
SMPL_PELVIS = 0
SMPL_LEFT_HIP = 1
SMPL_RIGHT_HIP = 2
SMPL_LEFT_KNEE = 4
SMPL_RIGHT_KNEE = 5
SMPL_LEFT_ANKLE = 7
SMPL_RIGHT_ANKLE = 8
SMPL_NECK = 12
SMPL_HEAD = 15
SMPL_LEFT_SHOULDER = 16
SMPL_RIGHT_SHOULDER = 17
SMPL_LEFT_ELBOW = 18
SMPL_RIGHT_ELBOW = 19
SMPL_LEFT_WRIST = 20
SMPL_RIGHT_WRIST = 21
NUM_SMPL_JOINTS = 24

# Our joint index → SMPL joint index.
_OURS_TO_SMPL: dict[int, int] = {
    int(Joint.PELVIS):         SMPL_PELVIS,
    int(Joint.LEFT_HIP):       SMPL_LEFT_HIP,
    int(Joint.RIGHT_HIP):      SMPL_RIGHT_HIP,
    int(Joint.LEFT_KNEE):      SMPL_LEFT_KNEE,
    int(Joint.RIGHT_KNEE):     SMPL_RIGHT_KNEE,
    int(Joint.LEFT_ANKLE):     SMPL_LEFT_ANKLE,
    int(Joint.RIGHT_ANKLE):    SMPL_RIGHT_ANKLE,
    int(Joint.NECK):           SMPL_NECK,
    int(Joint.HEAD):           SMPL_HEAD,
    int(Joint.LEFT_SHOULDER):  SMPL_LEFT_SHOULDER,
    int(Joint.RIGHT_SHOULDER): SMPL_RIGHT_SHOULDER,
    int(Joint.LEFT_ELBOW):     SMPL_LEFT_ELBOW,
    int(Joint.RIGHT_ELBOW):    SMPL_RIGHT_ELBOW,
    int(Joint.LEFT_WRIST):     SMPL_LEFT_WRIST,
    int(Joint.RIGHT_WRIST):    SMPL_RIGHT_WRIST,
}


@dataclass
class SMPLModel:
    v_template: Tensor       # (6890, 3)
    shapedirs: Tensor        # (6890, 3, 10)
    posedirs: Tensor         # (6890, 3, 207)
    J_regressor: Tensor      # (24, 6890)
    weights: Tensor          # (6890, 24)
    parents: Tensor          # (24,) int64
    faces: np.ndarray        # (13776, 3) int32

    @property
    def num_vertices(self) -> int:
        return int(self.v_template.shape[0])


SMPL_NPZ_DEFAULT = Path(__file__).resolve().parents[3] / "resources" / "smpl" / "SMPL_NEUTRAL.npz"


@lru_cache(maxsize=1)
def load_smpl(path: str | None = None, device: str = "cpu") -> SMPLModel:
    """Load the chumpy-free SMPL .npz and wrap it in tensors."""
    src = Path(path) if path else SMPL_NPZ_DEFAULT
    if not src.exists():
        raise FileNotFoundError(
            f"SMPL model not found at {src}. Run `uv run python scripts/setup_smpl.py` "
            f"after placing SMPL_NEUTRAL.pkl under resources/smpl/."
        )
    data = np.load(src, allow_pickle=False)
    dev = torch.device(device)

    def t(name: str, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.from_numpy(np.asarray(data[name])).to(device=dev, dtype=dtype)

    kintree = np.asarray(data["kintree_table"], dtype=np.int64)
    parents = kintree[0].copy()
    parents[0] = -1  # root marker
    return SMPLModel(
        v_template=t("v_template"),
        shapedirs=t("shapedirs"),
        posedirs=t("posedirs"),
        J_regressor=t("J_regressor"),
        weights=t("weights"),
        parents=torch.from_numpy(parents).to(dev),
        faces=np.asarray(data["f"], dtype=np.int32),
    )


# --------------------- Rotation utilities -----------------------------------


def matrix_to_axis_angle(R: Tensor) -> Tensor:
    """(*, 3, 3) → (*, 3) axis-angle. Standard log map on SO(3)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = cos_theta.arccos()
    sin_theta = theta.sin()
    # Small-angle: limit→0.5 of the skew-symmetric components.
    small = sin_theta.abs() < 1e-6
    sin_safe = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    rx = (R[..., 2, 1] - R[..., 1, 2]) / (2.0 * sin_safe)
    ry = (R[..., 0, 2] - R[..., 2, 0]) / (2.0 * sin_safe)
    rz = (R[..., 1, 0] - R[..., 0, 1]) / (2.0 * sin_safe)
    aa = torch.stack([rx, ry, rz], dim=-1) * theta.unsqueeze(-1)
    aa = torch.where(small.unsqueeze(-1), torch.zeros_like(aa), aa)
    return aa


def axis_angle_to_matrix(aa: Tensor) -> Tensor:
    """(*, 3) → (*, 3, 3) Rodrigues formula."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = aa / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    K = torch.stack(
        [
            torch.zeros_like(kx), -kz, ky,
            kz, torch.zeros_like(kx), -kx,
            -ky, kx, torch.zeros_like(kx),
        ],
        dim=-1,
    ).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    sin_t = theta.unsqueeze(-1).sin()
    cos_t = theta.unsqueeze(-1).cos()
    return eye + sin_t * K + (1 - cos_t) * (K @ K)


# --------------------- SMPL forward (LBS) -----------------------------------


def _smpl_pose_from_ours(rot_matrices: Tensor) -> Tensor:
    """Build a (B, 24, 3, 3) SMPL pose from our (B, 15, 3, 3) rotations.

    Unmapped SMPL joints (spine{1,2,3}, collars, hands, toes) stay at identity.
    """
    b = rot_matrices.shape[0]
    device, dtype = rot_matrices.device, rot_matrices.dtype
    smpl_R = torch.eye(3, device=device, dtype=dtype).expand(b, NUM_SMPL_JOINTS, 3, 3).clone()
    for ours_idx, smpl_idx in _OURS_TO_SMPL.items():
        smpl_R[:, smpl_idx] = rot_matrices[:, ours_idx]
    return smpl_R


def smpl_forward(
    smpl: SMPLModel,
    rot_matrices: Tensor,
    betas: Tensor | None = None,
) -> Tensor:
    """Return vertex positions (B, 6890, 3) for a batch of poses.

    ``rot_matrices``: (B, 15, 3, 3) — our skeleton's local rotations.
    """
    device, dtype = rot_matrices.device, rot_matrices.dtype
    b = rot_matrices.shape[0]
    if betas is None:
        betas = torch.zeros(b, smpl.shapedirs.shape[-1], device=device, dtype=dtype)

    # 1) shape blend
    v_shaped = smpl.v_template.unsqueeze(0) + torch.einsum("vds,bs->bvd", smpl.shapedirs, betas)

    # 2) joint locations
    J = torch.einsum("jv,bvd->bjd", smpl.J_regressor, v_shaped)  # (B, 24, 3)

    # 3) SMPL rotation matrices
    R = _smpl_pose_from_ours(rot_matrices)  # (B, 24, 3, 3)

    # 4) pose blend
    pose_feat = (R[:, 1:] - torch.eye(3, device=device, dtype=dtype)).reshape(b, -1)
    v_posed = v_shaped + torch.einsum("vdp,bp->bvd", smpl.posedirs, pose_feat)

    # 5) forward kinematics — accumulate 4×4 transforms in world space
    eye3 = torch.eye(3, device=device, dtype=dtype)
    parents = smpl.parents.tolist()

    def make_T(rot: Tensor, trans: Tensor) -> Tensor:
        T = torch.zeros((*rot.shape[:-2], 4, 4), device=device, dtype=dtype)
        T[..., :3, :3] = rot
        T[..., :3, 3] = trans
        T[..., 3, 3] = 1.0
        return T

    rel_joints = J.clone()
    rel_joints[:, 1:] = J[:, 1:] - J[:, parents[1:]]
    G_local = make_T(R, rel_joints)  # (B, 24, 4, 4)

    G_world: list[Tensor] = [None] * NUM_SMPL_JOINTS
    for j in range(NUM_SMPL_JOINTS):
        p = parents[j]
        G_world[j] = G_local[:, j] if p < 0 else G_world[p] @ G_local[:, j]
    G_world_t = torch.stack(G_world, dim=1)  # (B, 24, 4, 4)

    # Cancel the rest-pose offset so vertices align with v_template.
    J_pad = torch.cat([J, torch.zeros((b, NUM_SMPL_JOINTS, 1), device=device, dtype=dtype)], dim=-1)
    G_rel = G_world_t.clone()
    G_rel[..., :3, 3] = G_world_t[..., :3, 3] - torch.einsum("bjkl,bjl->bjk", G_world_t, J_pad)[..., :3]

    # 6) LBS — weighted blend of joint transforms
    T = torch.einsum("vj,bjkl->bvkl", smpl.weights, G_rel)
    v_homo = torch.cat(
        [v_posed, torch.ones((b, smpl.num_vertices, 1), device=device, dtype=dtype)], dim=-1
    )
    v_world = torch.einsum("bvkl,bvl->bvk", T, v_homo)[..., :3]
    return v_world


# --------------------- Helpers for visualization ----------------------------


def vertices_for_motion(
    rotations_6d: Tensor,
    root_translation: Tensor,
    smpl: SMPLModel,
    body_height_target: float = 1.0,
) -> Tensor:
    """Run SMPL on a whole sequence and return positioned vertices.

    rotations_6d   : (T, J=15, 6)
    root_translation : (T, 3) — added back after rescaling SMPL to our height.
    Returns        : (T, 6890, 3) tensor.
    """
    from wisp.data.kinematics import sixd_to_matrix

    rot_matrices = sixd_to_matrix(rotations_6d)  # (T, 15, 3, 3)
    verts = smpl_forward(smpl, rot_matrices)     # (T, 6890, 3)

    # Rescale: SMPL neutral body is ~1.7 m head-to-foot; we want body_height_target.
    rest_verts = smpl_forward(smpl, torch.eye(3, device=rot_matrices.device, dtype=rot_matrices.dtype)
                              .expand(1, NUM_JOINTS, 3, 3))[0]
    head_y = float(rest_verts[:, 1].max())
    foot_y = float(rest_verts[:, 1].min())
    scale = body_height_target / max(head_y - foot_y, 1e-6)
    verts = verts * scale

    # Anchor the SMPL pelvis (mean of hips region) to our root_translation.
    rest_J = (smpl.J_regressor @ smpl.v_template) * scale
    pelvis_offset = rest_J[SMPL_PELVIS].unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
    verts = verts - pelvis_offset + root_translation.unsqueeze(1)
    return verts
