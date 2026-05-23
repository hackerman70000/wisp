# wisp

Text-conditional motion diffusion for 15-joint stick figures. A
transformer-based DDPM trained on CMU mocap walks and jumps, generating
`[48, 15, 3]` skeleton tensors from prompts like _"a person is walking"_ or
_"the character jumps forward"_. Evaluated with FMD, MPJPE and inter-sample
variance.

|                  walk                   |                   jump                    |
| :-------------------------------------: | :---------------------------------------: |
| ![walk](outputs/report/sample_walk.gif) |  ![jump](outputs/report/sample_jump.gif)  |
|  _SMPL render, "a person is walking"_   | _SMPL render, "the character jumps forward"_ |

Four independent samples per class (skeleton view, same prompt, different seeds):

![walk grid](outputs/report/grid_walk.gif)
![jump grid](outputs/report/grid_jump.gif)

## Setup

Requires [uv](https://docs.astral.sh/uv/). macOS (MPS) and Linux (CUDA) are
both supported; on Linux CUDA wheels (`cu128`) are selected automatically.

```bash
uv sync
```

### SMPL body model (optional, for realistic rendering)

Single-sample GIFs can be rendered with a SMPL body mesh on top of the
predicted skeleton. Without it, the renderer falls back to the line + dot
stickman automatically.

1. Register at https://smpl.is.tue.mpg.de/ (free, research/non-commercial).
2. Download `SMPL_python_v.1.1.0.zip`.
3. Convert chumpy to numpy once:
   ```bash
   unzip -j SMPL_python_v.1.1.0.zip 'SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl' -d resources/smpl/
   mv resources/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl resources/smpl/SMPL_NEUTRAL.pkl
   uv run python scripts/setup_smpl.py
   ```

## Usage

```bash
uv run wisp download-data
uv run wisp prepare-data
uv run wisp train  --epochs 300 --batch-size 32
uv run wisp sample --prompt "a person is walking" --out walk.gif
uv run wisp sample --prompt "the character jumps forward" --out jump.gif
uv run wisp eval   --ckpt outputs/checkpoints/ema.pt
uv run wisp report --ckpt outputs/checkpoints/ema.pt
```

Defaults for `sample` / `report`: SMPL mesh, 200 DDIM steps, CFG=1.0,
Savitzky-Golay window 9. Override any with the respective flags.
