from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RESOURCES_DIR: Path = PROJECT_ROOT / "resources"
DEFAULT_DATASET_DIR: Path = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

NUM_JOINTS: int = 15
NUM_FRAMES: int = 48
TARGET_FPS: int = 24
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    rot_repr_dim: int = 6  # 6D continuous rotation
    root_dim: int = 3      # root translation
    cond_drop_prob: float = 0.1
    clip_arch: str = "ViT-B-32"
    clip_pretrained: str = "openai"


@dataclass(frozen=True)
class DiffusionConfig:
    num_train_timesteps: int = 1000
    num_sample_steps: int = 50
    beta_schedule: str = "cosine"
    cfg_scale: float = 2.5
    predict: str = "x0"  # MDM-style sample prediction


@dataclass(frozen=True)
class DataConfig:
    num_frames: int = NUM_FRAMES
    target_fps: int = TARGET_FPS
    augment_flip: bool = True
    augment_rotation: bool = True
    val_fraction: float = 0.1
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 400
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    ema_decay: float = 0.999
    lambda_pos: float = 1.0
    lambda_vel: float = 0.5
    lambda_accel: float = 0.0
    lambda_bone: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 10
    ckpt_every: int = 25
    num_workers: int = 2


@dataclass(frozen=True)
class WispConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
