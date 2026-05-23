"""3D stickman / SMPL body animation rendered as GIFs.

Coordinate convention: our motion tensor uses BVH-style **Y up**. Matplotlib's
3D axes use **Z up**, so when plotting we remap ``(x, y, z) → (x, z, y)``
— otherwise the figure appears lying on its side from a top-down angle.
The camera is set to a 3/4 side view (``elev=12, azim=-60``) which matches
the reference image in the project brief.

Two body modes are supported:
- ``skeleton`` (default fallback): bare line + dot stickman.
- ``smpl``: realistic SMPL mesh draped on top of the rotations. Requires
  the chumpy-free ``resources/smpl/SMPL_NEUTRAL.npz`` (see
  ``scripts/setup_smpl.py``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from wisp.data.skeleton import JOINT_CONNECTIONS

# 3/4 side view, slightly above the ground — matches the example figure
# in the project brief.
_VIEW_ELEV = 12.0
_VIEW_AZIM = -60.0


def _xyz_to_plot(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remap motion-space (X, Y-up, Z) to matplotlib (X, Z-into, Y-up)."""
    return frame[..., 0], frame[..., 2], frame[..., 1]


def _setup_axis(ax, motion: np.ndarray, title: str = "", small: bool = False) -> None:
    xs, ys, zs = _xyz_to_plot(motion)
    cx = (float(xs.max()) + float(xs.min())) / 2
    cy = (float(ys.max()) + float(ys.min())) / 2
    cz = (float(zs.max()) + float(zs.min())) / 2
    half = max(
        float(xs.max()) - float(xs.min()),
        float(ys.max()) - float(ys.min()),
        float(zs.max()) - float(zs.min()),
    ) / 2 + 0.15
    half = max(half, 0.8)  # don't shrink below a stickman-sized box

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=_VIEW_ELEV, azim=_VIEW_AZIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if title:
        ax.set_title(title, fontsize=9 if small else 11)


def animate_skeleton_3d(
    tensor_data: np.ndarray,
    output_path: Path | None = None,
    fps: int = 24,
    title: str = "",
    show: bool = False,
) -> None:
    """Render the line + dot stickman from ``(T, 15, 3)`` positions."""
    if tensor_data.ndim != 3 or tensor_data.shape[1] != 15 or tensor_data.shape[2] != 3:
        raise ValueError(f"expected shape (T, 15, 3); got {tensor_data.shape}")

    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    _setup_axis(ax, tensor_data, title=title)

    points_scatter = ax.scatter([], [], [], c="#ef4444", s=40, zorder=3)
    lines = [ax.plot([], [], [], c="#3b82f6", lw=2.0, zorder=2)[0] for _ in JOINT_CONNECTIONS]

    def update(frame_idx: int):
        frame = tensor_data[frame_idx]
        xs, ys, zs = _xyz_to_plot(frame)
        points_scatter._offsets3d = (xs, ys, zs)
        for line_obj, (a, b) in zip(lines, JOINT_CONNECTIONS, strict=True):
            seg = np.stack([frame[a], frame[b]])
            sx, sy, sz = _xyz_to_plot(seg)
            line_obj.set_data(sx, sy)
            line_obj.set_3d_properties(sz)
        return [points_scatter, *lines]

    anim = animation.FuncAnimation(
        fig, update, frames=tensor_data.shape[0], blit=False, interval=1000 / fps,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(output_path), writer="pillow", fps=fps)
    if show:
        plt.show()
    plt.close(fig)


def _shade_triangles(verts: np.ndarray, faces: np.ndarray, light_dir, ambient: float = 0.35,
                     base=(0.95, 0.78, 0.65)) -> np.ndarray:
    """Per-face Lambertian shading. ``verts``: (V, 3), ``faces``: (F, 3) int."""
    tri = verts[faces]                      # (F, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = n_norm.squeeze(-1) > 1e-9
    n_unit = np.where(n_norm > 1e-9, normals / np.where(n_norm > 1e-9, n_norm, 1.0), 0.0)
    dotp = n_unit @ np.asarray(light_dir)
    intensity = np.where(valid, ambient + (1.0 - ambient) * np.clip(dotp, 0.0, 1.0), ambient)
    base_arr = np.asarray(base, dtype=np.float32)
    colors = np.clip(intensity[:, None] * base_arr[None], 0.0, 1.0)
    return np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.float32)], axis=-1)


def _remap_xyz(verts: np.ndarray) -> np.ndarray:
    """Apply the Y-up → Z-up swap used by ``_xyz_to_plot`` to a vertex array."""
    out = np.empty_like(verts)
    out[..., 0] = verts[..., 0]
    out[..., 1] = verts[..., 2]
    out[..., 2] = verts[..., 1]
    return out


def animate_smpl_3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path | None = None,
    fps: int = 24,
    title: str = "",
    show: bool = False,
) -> None:
    """Render a SMPL mesh animation from ``(T, V, 3)`` vertices + ``(F, 3)`` faces."""
    if vertices.ndim != 3:
        raise ValueError(f"expected vertices shape (T, V, 3); got {vertices.shape}")

    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    _setup_axis(ax, vertices.reshape(-1, 15, 3) if vertices.shape[1] == 15
                else _proxy_bbox(vertices), title=title)

    light_dir = np.array([0.5, 0.7, 0.6])
    light_dir = light_dir / np.linalg.norm(light_dir)
    body_coll = Poly3DCollection([], edgecolor=None, linewidth=0)
    ax.add_collection3d(body_coll)

    def update(frame_idx: int):
        verts = vertices[frame_idx]
        plot_verts = _remap_xyz(verts)
        tri_verts = plot_verts[faces]      # (F, 3, 3)
        body_coll.set_verts(tri_verts)
        body_coll.set_facecolor(_shade_triangles(verts, faces, light_dir))
        return [body_coll]

    anim = animation.FuncAnimation(
        fig, update, frames=vertices.shape[0], blit=False, interval=1000 / fps,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(output_path), writer="pillow", fps=fps)
    if show:
        plt.show()
    plt.close(fig)


def _proxy_bbox(vertices: np.ndarray) -> np.ndarray:
    """Mock a (T, 15, 3) joint trajectory from a (T, V, 3) vertex cloud so
    ``_setup_axis`` can use its existing bbox logic without modification."""
    t = vertices.shape[0]
    # Sample 15 vertices roughly spanning the bbox per frame.
    idx = np.linspace(0, vertices.shape[1] - 1, 15, dtype=np.int64)
    return vertices[:, idx, :]


def animate_grid(
    motions: list[np.ndarray],
    output_path: Path,
    titles: list[str] | None = None,
    fps: int = 24,
    cols: int = 4,
) -> None:
    """Render a grid of skeleton animations into a single GIF.

    For per-sample comparison only the line + dot stickman is used —
    rendering 4× SMPL meshes per frame with matplotlib is too slow to be
    worth it.
    """
    n = len(motions)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(2.8 * cols, 2.8 * rows), dpi=80)

    scatters, lines_per_ax = [], []
    for i, motion in enumerate(motions):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _setup_axis(ax, motion, title=titles[i] if titles else "", small=True)
        scatters.append(ax.scatter([], [], [], c="#ef4444", s=20, zorder=3))
        lines = [ax.plot([], [], [], c="#3b82f6", lw=1.5, zorder=2)[0] for _ in JOINT_CONNECTIONS]
        lines_per_ax.append(lines)

    num_frames = motions[0].shape[0]

    def update(frame_idx: int):
        artists = []
        for motion, scatter, lines in zip(motions, scatters, lines_per_ax, strict=True):
            frame = motion[frame_idx]
            xs, ys, zs = _xyz_to_plot(frame)
            scatter._offsets3d = (xs, ys, zs)
            artists.append(scatter)
            for line_obj, (a, b) in zip(lines, JOINT_CONNECTIONS, strict=True):
                seg = np.stack([frame[a], frame[b]])
                sx, sy, sz = _xyz_to_plot(seg)
                line_obj.set_data(sx, sy)
                line_obj.set_3d_properties(sz)
                artists.append(line_obj)
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, blit=False, interval=1000 / fps
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)
