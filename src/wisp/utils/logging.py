import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path | None = None, run_name: str = "run", level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / f"{run_name}.log",
            level="DEBUG",
            rotation="100 MB",
            retention="14 days",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )


def get_device(preferred: str | None = None) -> str:
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log_environment(device: str) -> None:
    import torch

    logger.info(f"torch {torch.__version__} | device={device}")
    if device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        logger.info(f"cuda: {name} (sm_{major}{minor}) | {vram:.1f} GB")
