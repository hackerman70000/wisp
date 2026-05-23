"""Convert SMPL_NEUTRAL.pkl (chumpy-encoded, from smpl.is.tue.mpg.de) into a
plain numpy .npz so the rest of the project does not depend on chumpy.

Usage
-----
    1. Register at https://smpl.is.tue.mpg.de/, download SMPL_python_v.1.1.0.zip.
    2. Extract ``models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl``.
    3. Rename to ``SMPL_NEUTRAL.pkl`` and place at ``resources/smpl/SMPL_NEUTRAL.pkl``.
    4. Run: ``uv run python scripts/setup_smpl.py``

The resulting ``resources/smpl/SMPL_NEUTRAL.npz`` is loaded by
``wisp.viz.smpl_renderer``.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
SMPL_PKL = ROOT / "resources" / "smpl" / "SMPL_NEUTRAL.pkl"
SMPL_NPZ = ROOT / "resources" / "smpl" / "SMPL_NEUTRAL.npz"


def _stub_chumpy() -> None:
    """Install a minimal fake ``chumpy`` so pickle can resurrect Ch objects.

    Chumpy stores its array in ``self.x`` (numpy ndarray); we don't need any
    of the symbolic-differentiation machinery — just to pull ``x`` back out.
    """
    if "chumpy" in sys.modules:
        return

    class _ChStub:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["_state"] = state

        def to_array(self):
            for key in ("x", "_x", "a", "r"):
                if key in self.__dict__:
                    return np.asarray(self.__dict__[key])
            return None

    mod = types.ModuleType("chumpy")
    ch_mod = types.ModuleType("chumpy.ch")
    ch_mod.Ch = _ChStub
    mod.Ch = _ChStub
    mod.ch = ch_mod
    sys.modules["chumpy"] = mod
    sys.modules["chumpy.ch"] = ch_mod


def _to_numpy(value):
    """Convert a value (possibly a chumpy stub) into a plain numpy array."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "to_array"):
        arr = value.to_array()
        if arr is not None:
            return arr
    return None


def main() -> int:
    if not SMPL_PKL.exists():
        logger.error(f"SMPL_NEUTRAL.pkl not found at {SMPL_PKL}")
        logger.error("Download it from https://smpl.is.tue.mpg.de/ (see docstring).")
        return 1

    _stub_chumpy()
    with SMPL_PKL.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    keep = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights",
            "kintree_table", "f")
    out: dict[str, np.ndarray] = {}
    for key in keep:
        if key not in data:
            continue
        value = data[key]
        arr = _to_numpy(value)
        if arr is None:
            arr = value
        if hasattr(arr, "toarray"):  # scipy sparse matrix (J_regressor)
            arr = arr.toarray()
        arr = np.asarray(arr, dtype=np.float32 if key != "kintree_table" and key != "f"
                         else (np.int64 if key == "kintree_table" else np.int32))
        out[key] = arr
        logger.info(f"  {key}: {arr.shape} {arr.dtype}")

    # SMPL ships 300 shape blend shapes (10 body shape + 290 hand pose). Only the
    # first 10 correspond to body shape parameters used at inference.
    if out.get("shapedirs", np.zeros((0, 0, 0))).shape[-1] > 10:
        out["shapedirs"] = out["shapedirs"][..., :10]
        logger.info(f"  shapedirs sliced to first 10: {out['shapedirs'].shape}")

    expected = {"v_template", "shapedirs", "posedirs", "J_regressor", "weights",
                "kintree_table", "f"}
    missing = expected - out.keys()
    if missing:
        logger.error(f"missing keys after conversion: {missing}")
        return 2

    SMPL_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SMPL_NPZ, **out)
    logger.success(
        f"wrote {SMPL_NPZ} ({SMPL_NPZ.stat().st_size / 1e6:.1f} MB) "
        f"with keys: {sorted(out)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
