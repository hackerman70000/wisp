"""Download the curated walk / jump subset from the CMU mocap mirror."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wisp.data.catalog import CLIPS_BY_LABEL, bvh_url, clip_name  # noqa: E402
from wisp.utils.logging import setup_logging  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "raw"


def download(url: str, dst: Path, retries: int = 3, backoff: float = 1.5) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            dst.write_bytes(response.content)
            return True
        except requests.RequestException as exc:
            logger.warning(f"{dst.name} attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
    logger.error(f"giving up on {url}")
    return False


def main() -> None:
    setup_logging(run_name="download")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    total = sum(len(clips) for clips in CLIPS_BY_LABEL.values())
    done = 0
    for label, clips in CLIPS_BY_LABEL.items():
        label_dir = DATA_ROOT / label
        label_dir.mkdir(exist_ok=True)
        for subject, trial in clips:
            dst = label_dir / f"{clip_name(subject, trial)}.bvh"
            ok = download(bvh_url(subject, trial), dst)
            done += 1
            status = "ok" if ok else "fail"
            logger.info(f"[{done:>3}/{total}] {label}/{dst.name} {status}")

    sizes = {label: len(list((DATA_ROOT / label).glob("*.bvh"))) for label in CLIPS_BY_LABEL}
    logger.success(f"done — {sizes}")


if __name__ == "__main__":
    main()
