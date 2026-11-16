# Running Inference from `inference.ipynb`

Step-by-step instructions for running inference of the ISP pipeline (RAW → JDD/NAFNet → … → HDRNet) from the `inference.ipynb` notebook.

## 1. Contents and Requirements

### What must be present in the project
- `inference.ipynb` — the inference notebook.
- `src/` — pipeline code (`blocks/`, `nafnet/`, `hdrnet/`, `metrics.py`, `utils.py`).
- `checkpoints/jdd/nafnet_jdd_epoch_175.pt` — JDD (NAFNet) weights.
- `checkpoints/hdrnet/epoch30.pth` and `checkpoints/hdrnet/config.json` — HDRNet weights and config.
- `requirements.txt` — dependencies.

### Environment
- **Python**: 3.13.x is recommended (the pipeline was validated on Python 3.13.3, macOS arm64, CPU).
- **Device**: the notebook uses CPU by default (`DEVICE = torch.device("cpu")`).

> **Important:** the notebook must be run from the **project root** (`PACKAGE_DIR = os.path.abspath(".")`, imports like `from src...`). Running it from a different working directory will cause path and import errors.

## 2. Data Preparation

### Input data (RAW)
In the input data folder (by default `data/raw/`), each frame must have two files with the same index `i`:

- `{i}.png` — 16-bit grayscale RAW Bayer PNG (expected size 6144×8192).
- `{i}.json` — frame metadata.

Example `data/raw/900.json`:
```json
{
  "black_level": [256, 256, 256, 256],
  "white_level": 4095,
  "noise_profile": [0.003454695275, 5.14075368211699e-05],
  "cfa_pattern": [0, 1, 1, 2],
  "orientation": "Rotate 90 CW",
  "as_shot_neutral": [0.3355869699, 1.0, 0.3828892006],
  "huawei_bounds": [658, 2963, 0, 2654]
}
```

### Ground truth (optional, for metrics)
In the GT folder (by default `data/ground_truth/`) — `{i}.JPG` files (2000×2000); names must match the RAW frame indices.  
If GT is missing for a frame or its size does not match the result, inference will still run, but PSNR/SSIM for that frame will be skipped.

### Results
The output folder (by default `output/`) is created automatically. It stores:
- `{i}.png` — final images;
- `metrics.csv` — table with columns `frame, psnr, ssim, seconds` (if metrics saving is enabled).

## 3. Environment Setup

Run all commands from the repository root:

```bash
cd NTIRE_2025_NPR

# 1) Create a virtual environment
python3.13 -m venv .venv

# 2) Activate the environment
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Running the Notebook

1. Open `inference.ipynb`.
2. Select the **kernel/interpreter** of the created environment (`.venv`).

## 5. Configuring Parameters ("Run Configuration" cell)

Before running, set the parameters in the configuration cell:

| Parameter | Description | Default value |
|---|---|---|
| `INPUT_DIR` | Folder with `{i}.png` + `{i}.json` pairs | `data/raw` |
| `GT_DIR` | Folder with ground truth `{i}.JPG` (for PSNR/SSIM) | `data/ground_truth` |
| `OUTPUT_DIR` | Folder for saving PNGs and `metrics.csv` | `output` |
| `FRAME_RANGE` | `None` — all frames found in `INPUT_DIR`; `(start, end)` — inclusive index range | `None` |
| `SAVE_IMAGES` | Save final PNGs | `True` |
| `COMPUTE_METRICS` | Compute PSNR/SSIM against GT | `True` |
| `SKIP_EXISTING` | Skip frames whose PNG already exists in `OUTPUT_DIR` (when `SAVE_IMAGES=True`) | `False` |
| `SAVE_METRICS_CSV` | Save `metrics.csv` to `OUTPUT_DIR` | `True` |

Examples:
- Process all frames from `INPUT_DIR`: `FRAME_RANGE = None`
- Process only frame 900: `FRAME_RANGE = (900, 900)`
- Process a range: `FRAME_RANGE = (100, 200)`
- Inference only, without metrics: `COMPUTE_METRICS = False` (and/or an empty/missing `GT_DIR`)

## 6. Cell Execution Order

It is recommended to run **Run All** from top to bottom. Cell purposes:

1. **Imports and setup** — adds the project root to `sys.path`, imports modules (`src.blocks...`, `src.nafnet...`, `src.hdrnet...`, metrics), sets `DEVICE = cpu`, and prints `torch` version, device, and `PACKAGE_DIR`.
2. **Run Configuration** — applies paths/flags, builds the `FRAMES` list, creates `OUTPUT_DIR` if needed, and prints the configuration and found frames.
3. **JDD Weights (NAFNet)** — loads the JDD model and the `checkpoints/jdd/nafnet_jdd_epoch_175.pt` checkpoint, and switches the model to `eval()` mode.
4. **HDRNet Weights** — loads parameters/weights from `checkpoints/hdrnet/epoch30.pth`, initializes `HDRPointwiseNN`, and switches it to `eval()` mode.
5. **Color and Geometry Matrices** — defines transformation matrices (`COLOR_MATRIX`, `XYZ2SRGB`, `TRANSFORMATION_MATRIX`) and tone mapping parameters.
6. **Pipeline Construction** — assembles the processing chain:
   `InputRaw → BlackLevelCorrection → Normalization → JDD → AutoWhiteBalance → XYZTransform → SRGBTransform → UnrealToneMapping → GammaCorrection → SpatialTransform → HDRNet`.
7. **Single Frame: GT vs Prediction** — runs inference for one frame (`SAMPLE_INDEX = FRAMES[0]`), prints the time, computes PSNR/SSIM if GT is available, and shows a visual comparison. Useful for a quick sanity check.
8. **Batch Run: PNG saving and PSNR/SSIM** — processes all frames in `FRAMES`, saves results, computes metrics, and produces a final summary.

## 7. Results

- **Images:** `OUTPUT_DIR/{i}.png` — final pipeline frames.
- **Metrics:** `OUTPUT_DIR/metrics.csv` with columns `frame, psnr, ssim, seconds` (if `SAVE_METRICS_CSV=True`).
- **Notebook output:**
  - information about loaded weights and parameters;
  - time and (if GT is available) PSNR/SSIM for the sample frame and visualization;
  - a per-frame table;
  - summary PSNR/SSIM statistics (`mean/std/min/max`) and output paths.
