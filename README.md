# NTIRE_2025_NPR: ISP Pipeline Inference

Delivery package for inference of the night photo-rendering pipeline with a **JDD block**
(joint demosaic + denoise, NAFNet) and **HDRNet**.

## Pipeline

```
RAW (16-bit Bayer PNG + JSON, 6144×8192)
 -> BlackLevelCorrection
 -> Normalization
 -> JDD (NAFNet)                # joint demosaic + denoise
 -> AutoWhiteBalance            # as_shot_neutral
 -> XYZTransform
 -> SRGBTransform
 -> UnrealToneMapping
 -> GammaCorrection
 -> SpatialTransform
 -> HDRNet
 -> Output (2000x2000)
```

## Package Contents

```
NTIRE_2025_NPR/
├── README.md               # this file
├── requirements.txt        # dependencies
├── inference.ipynb         # inference notebook
├── src/                    # pipeline blocks, NAFNet, HDRNet, metrics
│   ├── blocks/             # pipeline blocks
│   ├── nafnet/             # NAFNet (JDD): model, wrapper, checkpoint loading
│   ├── hdrnet/             # HDRNet (HDRPointwiseNN) + bilateral slice
│   ├── metrics.py          # PSNR / SSIM metrics
│   └── utils.py            # helpers (JSON reading, etc.)
├── checkpoints/
│   ├── jdd/nafnet_jdd_epoch_175.pt      # JDD (NAFNet) weights
│   └── hdrnet/epoch30.pth + config.json # HDRNet weights
└── data/                   # demo data: frame 900
    ├── raw/                # {i}.png + {i}.json
    └── ground_truth/       # {i}.JPG
```

Weights used:

| Model | File |
|---|---|
| JDD (NAFNet) | `checkpoints/jdd/nafnet_jdd_epoch_175.pt` |
| HDRNet | `checkpoints/hdrnet/epoch30.pth` |

## Installation

Tested on Python 3.13.3 (macOS arm64, CPU inference).

```bash
cd NTIRE_2025_NPR
python3.13 -m venv delivery_venv
source delivery_venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
jupyter lab inference.ipynb
# or open inference.ipynb in VS Code with the delivery_venv kernel
```

In the notebook, in the "Run Configuration" cell:

- `INPUT_DIR` — path to input images: a folder with `{i}.png` + `{i}.json` pairs;
- `GT_DIR` — folder with ground truth `{i}.JPG` (needed only for PSNR/SSIM metrics;
  may be missing — metrics for frames without GT are skipped);
- `OUTPUT_DIR` — where to save the results;
- `FRAME_RANGE` — `None` (all found frames) or an inclusive `(start, end)` range.

Run results:

- final PNGs `{i}.png` in `OUTPUT_DIR`;
- `metrics.csv` (frame, psnr, ssim, seconds) in `OUTPUT_DIR`;
- a PSNR/SSIM table and summary statistics (mean/std/min/max) in the cell outputs.

## Input Data Format

- `{i}.png` — 16-bit grayscale RAW PNG (6144×8192);
- `{i}.json` — frame metadata with the fields:

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

- ground truth — `{i}.JPG` (2000×2000); names must match the raw frames.
