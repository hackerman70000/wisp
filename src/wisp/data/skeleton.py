"""15-joint stickman skeleton — joint indices, hierarchy, BVH mapping."""

from __future__ import annotations

from enum import IntEnum


class Joint(IntEnum):
    HEAD = 0
    NECK = 1
    PELVIS = 2
    RIGHT_SHOULDER = 3
    RIGHT_ELBOW = 4
    RIGHT_WRIST = 5
    LEFT_SHOULDER = 6
    LEFT_ELBOW = 7
    LEFT_WRIST = 8
    RIGHT_HIP = 9
    RIGHT_KNEE = 10
    RIGHT_ANKLE = 11
    LEFT_HIP = 12
    LEFT_KNEE = 13
    LEFT_ANKLE = 14


NUM_JOINTS = len(Joint)

# Parent of each joint; root (PELVIS) is -1.
PARENTS: tuple[int, ...] = (
    Joint.NECK,           # HEAD
    Joint.PELVIS,         # NECK
    -1,                   # PELVIS (root)
    Joint.NECK,           # RIGHT_SHOULDER
    Joint.RIGHT_SHOULDER, # RIGHT_ELBOW
    Joint.RIGHT_ELBOW,    # RIGHT_WRIST
    Joint.NECK,           # LEFT_SHOULDER
    Joint.LEFT_SHOULDER,  # LEFT_ELBOW
    Joint.LEFT_ELBOW,     # LEFT_WRIST
    Joint.PELVIS,         # RIGHT_HIP
    Joint.RIGHT_HIP,      # RIGHT_KNEE
    Joint.RIGHT_KNEE,     # RIGHT_ANKLE
    Joint.PELVIS,         # LEFT_HIP
    Joint.LEFT_HIP,       # LEFT_KNEE
    Joint.LEFT_KNEE,      # LEFT_ANKLE
)

# Edges for visualization (parent → child).
JOINT_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (Joint.PELVIS, Joint.NECK),
    (Joint.NECK, Joint.HEAD),
    (Joint.NECK, Joint.RIGHT_SHOULDER),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.NECK, Joint.LEFT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.PELVIS, Joint.RIGHT_HIP),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    (Joint.PELVIS, Joint.LEFT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
)

# CMU BVH joint names that map to each of our 15 joints. The CMU rig has
# extra spine / hand joints we collapse away — only the world position of
# these joints is read.
BVH_JOINT_NAMES: dict[int, str] = {
    Joint.HEAD: "Head",
    Joint.NECK: "Neck1",
    Joint.PELVIS: "Hips",
    Joint.RIGHT_SHOULDER: "RightArm",
    Joint.RIGHT_ELBOW: "RightForeArm",
    Joint.RIGHT_WRIST: "RightHand",
    Joint.LEFT_SHOULDER: "LeftArm",
    Joint.LEFT_ELBOW: "LeftForeArm",
    Joint.LEFT_WRIST: "LeftHand",
    Joint.RIGHT_HIP: "RightUpLeg",
    Joint.RIGHT_KNEE: "RightLeg",
    Joint.RIGHT_ANKLE: "RightFoot",
    Joint.LEFT_HIP: "LeftUpLeg",
    Joint.LEFT_KNEE: "LeftLeg",
    Joint.LEFT_ANKLE: "LeftFoot",
}

# Joint pairs swapped when mirroring the skeleton across the sagittal plane.
LR_SWAP: tuple[tuple[int, int], ...] = (
    (Joint.RIGHT_SHOULDER, Joint.LEFT_SHOULDER),
    (Joint.RIGHT_ELBOW, Joint.LEFT_ELBOW),
    (Joint.RIGHT_WRIST, Joint.LEFT_WRIST),
    (Joint.RIGHT_HIP, Joint.LEFT_HIP),
    (Joint.RIGHT_KNEE, Joint.LEFT_KNEE),
    (Joint.RIGHT_ANKLE, Joint.LEFT_ANKLE),
)
