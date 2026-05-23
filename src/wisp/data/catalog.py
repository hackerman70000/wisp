"""Curated CMU mocap clip list for walks and jumps.

Each entry is (subject, trial). Source repository:
    https://github.com/una-dinosauria/cmu-mocap

Selections are filtered from the official index (`cmu-mocap-index-text.txt`)
to keep clean, single-action recordings — pure walks (no turn / no stop) and
single-jump captures (forward / standing / high jumps). Mixed clips
("walk, sit, walk") are excluded to keep the conditioning signal tight.
"""

from __future__ import annotations

WALK_CLIPS: tuple[tuple[int, int], ...] = (
    (2, 1), (2, 2),
    (7, 1), (7, 2), (7, 3), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11),
    (8, 1), (8, 2), (8, 3), (8, 6), (8, 8), (8, 9), (8, 10),
    (12, 1), (12, 2), (12, 3),
    (16, 15), (16, 16), (16, 21), (16, 31), (16, 47), (16, 58),
    (26, 1),
    (27, 1),
    (29, 1),
    (32, 1), (32, 2),
    (35, 1), (35, 2), (35, 3), (35, 4),
    (37, 1),
    (38, 1), (38, 2), (38, 4),
    (39, 1), (39, 2), (39, 3), (39, 4), (39, 5), (39, 6), (39, 7), (39, 8),
    (39, 9), (39, 10), (39, 11), (39, 12), (39, 13), (39, 14),
    (45, 1),
    (47, 1),
    (49, 1), (49, 2), (49, 3),
    (55, 4),
    (56, 1),
    (69, 1), (69, 2),
)

JUMP_CLIPS: tuple[tuple[int, int], ...] = (
    (13, 11), (13, 13), (13, 19), (13, 32), (13, 39), (13, 40), (13, 41), (13, 42),
    (16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 9), (16, 10),
    (49, 4), (49, 5),
    (75, 1), (75, 2), (75, 3), (75, 4), (75, 5), (75, 6), (75, 7), (75, 8),
    (75, 9), (75, 10), (75, 11), (75, 12), (75, 13), (75, 14), (75, 15),
    (118, 1), (118, 2), (118, 3), (118, 4), (118, 5), (118, 6), (118, 7),
    (118, 8), (118, 9), (118, 10),
    (127, 4), (127, 5), (127, 6),
)


def bvh_url(subject: int, trial: int) -> str:
    """Raw GitHub URL for a CMU mocap BVH file (una-dinosauria mirror)."""
    return (
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/"
        f"data/{subject:03d}/{subject:02d}_{trial:02d}.bvh"
    )


def clip_name(subject: int, trial: int) -> str:
    return f"{subject:02d}_{trial:02d}"


CLIPS_BY_LABEL: dict[str, tuple[tuple[int, int], ...]] = {
    "walk": WALK_CLIPS,
    "jump": JUMP_CLIPS,
}
