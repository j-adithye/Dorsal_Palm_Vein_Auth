# CNN-Based Dorsal Hand Vein Authentication Using Triplet Loss Metric Learning

> **Published in IJARCCE · Vol. 15, Issue 3, March 2026**  
> 📄 [https://doi.org/10.17148/IJARCCE.2026.15372](https://doi.org/10.17148/IJARCCE.2026.15372)

A complete end-to-end biometric authentication system that recognises individuals from their **dorsal hand vein patterns** captured under near-infrared (NIR) illumination. The system runs fully on a **Raspberry Pi 4B** — no cloud, no internet required.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Results](#results)
4. [Repository Structure](#repository-structure)
5. [Hardware Setup](#hardware-setup)
6. [Dataset](#dataset)
7. [Preprocessing Pipeline](#preprocessing-pipeline)
8. [ML Model](#ml-model)
9. [On-Device Deployment (`auth_on_pi`)](#on-device-deployment-auth_on_pi)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Configuration Reference](#configuration-reference)
13. [Authors](#authors)
14. [Citation](#citation)

---

## Overview

Hand vein biometrics offer strong security because vascular patterns lie beneath the skin and are extremely difficult to spoof. This project implements a full biometric authentication pipeline:

- **Near-infrared image capture** using an OV5647 NoIR camera + 850 nm LEDs
- **Ten-stage preprocessing pipeline** to isolate the stable metacarpal vein region
- **Four-block CNN backbone** trained with **online semi-hard triplet loss mining** to generate 128-dimensional L2-normalised embeddings
- **Edge deployment** on Raspberry Pi 4B with TFLite INT8 inference
- **Flask web app + CLI** for registration and verification

---

## System Architecture

```
NIR Camera (OV5647 + 850nm LEDs)
         │
         ▼
┌─────────────────────────┐
│  10-Stage Preprocessing │  Gaussian blur → Smart crop → Wrist removal →
│  Pipeline (preprocessing│  Finger removal → Otsu mask → CLAHE →
│  .py)                   │  Sato vesselness → Feathered mask → Crop →
└─────────────────────────┘  Z-score normalise → 224×224×1 image
         │
         ▼
┌─────────────────────────┐
│  CNN Backbone (TFLite)  │  4 conv blocks (32→64→128→256 filters)
│  128-dim L2 embedding   │  → GlobalAvgPool → Dense 256 → Dense 128
└─────────────────────────┘  → L2 normalise
         │
         ▼
┌─────────────────────────┐
│  Euclidean Distance     │  Compare query embedding against
│  Matching + Threshold   │  stored enrollment templates (SQLite)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Liveness Check         │  MLX90614 measures hand surface temp
│  (MLX90614 sensor)      │  Valid range: 30°C – 37°C
└─────────────────────────┘
         │
         ▼
    ACCEPT / REJECT
```

---

## Results

| Metric                                   | Value                           |
| ---------------------------------------- | ------------------------------- |
| Recognition Accuracy                     | **99.12%** (threshold = 0.5371) |
| Equal Error Rate (EER)                   | **1.32%** (threshold = 0.6052)  |
| FAR @ EER                                | 1.32%                           |
| FRR @ EER                                | 1.32%                           |
| Authentication threshold (normal)        | 0.30                            |
| Authentication threshold (high security) | 0.25                            |
| TFLite INT8 inference time (Pi 4B)       | < 200 ms                        |
| Preprocessing time (Pi 4B)               | < 1800 ms                       |
| Full pipeline latency                    | < 2 second per query            |

Training used the Adam optimiser with an initial learning rate of 1×10⁻⁴, cosine annealing with 5-epoch warm-up, triplet loss margin of 1.5, and a maximum of 50 epochs (early stopping patience 15). Batch size was 128 (K=32 identities × N=4 images).

---

## Repository Structure

```
Dorsal_Palm_Vein_Auth/
│
├── Preprocessing/
│   ├── preprocessing.py        # Full 10-stage NIR preprocessing pipeline
│   └── converter.py            # Batch parallel dataset preprocessing tool
│
├── ml model/
│   ├── dorsal_palm_vein_model.py  # CNN + triplet loss training script (Kaggle)
│   └── augmentation.py            # Data augmentation (rotation, noise, grid distortion …)
│
├── auth_on_pi/                 # Complete on-device authentication system
│   ├── app.py                  # Flask web application (MJPEG stream + REST API)
│   ├── auth.py                 # Registration / verification / identification logic
│   ├── inference.py            # TFLite model loading + preprocessing + embedding
│   ├── camera.py               # Picamera2 capture module (OV5647 NoIR)
│   ├── embeddings.py           # SQLite embedding storage (CRUD)
│   ├── config.py               # Central configuration (all tuneable parameters)
│   ├── cli.py                  # Terminal interface for 1:1 verification
│   ├── preprocessing.py        # Pipeline copy used at inference time
│   ├── models/
│   │   ├── cnn_backbone.tflite     # Base TFLite model
│   │   └── cnn_backbone_ft.tflite  # Fine-tuned TFLite model (recommended)
│   ├── data/
│   │   └── embeddings.db       # SQLite database (auto-created on first run)
│   ├── static/
│   │   ├── app.js
│   │   └── style.css
│   └── templates/
│       ├── index.html
│       ├── register.html
│       ├── verify.html
│       ├── identify.html
│       ├── admin.html
│       ├── debug.html
│       └── base.html
│
├── tflite/
│   └── cnn_backbone_ft.tflite  # Standalone fine-tuned TFLite model
│
└── result/
    ├── far_frr_eer_curve.png   # FAR/FRR/EER evaluation plot
    └── training_curves.png     # Training & validation triplet loss curves
```

---

## Hardware Setup

| Component       | Details                                                  |
| --------------- | -------------------------------------------------------- |
| Processing unit | Raspberry Pi 4 Model B (4 GB RAM)                        |
| Camera          | Arducam OV5647 5MP NoIR (no IR-cut filter)               |
| Illumination    | 850 nm near-infrared LEDs                                |
| Liveness sensor | MLX90614 infrared temperature sensor (I2C, address 0x5A) |

**Camera parameters (fixed, no auto-exposure):**

| Parameter          | Value                             |
| ------------------ | --------------------------------- |
| AWB gains (R, B)   | 1.0, 1.0                          |
| Shutter speed      | 19 000 µs                         |
| Analogue gain      | 1.1                               |
| Contrast           | 1.6                               |
| Capture resolution | 1296 × 972 px (OV5647 native 4:3) |

The MLX90614 connects to the Pi via I2C bus 1 (SDA = GPIO 2, SCL = GPIO 3). Pull-up resistors (4.7 kΩ) are required on both lines. The sensor rejects any sample whose surface temperature falls outside 30°C – 37°C, blocking spoofing attacks using printed images or artificial replicas.

---

## Dataset

A custom dataset was collected specifically for this project because publicly available hand vein datasets are typically captured under different and inconsistent conditions.

| Property             | Value                                                            |
| -------------------- | ---------------------------------------------------------------- |
| Total participants   | 261 individuals                                                  |
| Biometric identities | 522 (left and right hands treated separately)                    |
| Images per identity  | 10                                                               |
| Total images         | 5 220                                                            |
| Imaging modality     | Near-infrared (850 nm)                                           |
| Image content        | Full dorsal hand (fingers + wrist included before preprocessing) |

**Train / Validation / Test split (identity-level, no leakage):**

| Subset     | Identities | Purpose                                |
| ---------- | ---------- | -------------------------------------- |
| Train      | ~391 (75%) | Model training                         |
| Validation | ~78 (15%)  | Threshold calibration + early stopping |
| Test       | ~53 (10%)  | Final evaluation                       |

Identity-level splitting guarantees that no images of the same individual appear in both training and evaluation sets.

---

## Preprocessing Pipeline

The 10-stage pipeline is implemented in `Preprocessing/preprocessing.py` and mirrored in `auth_on_pi/preprocessing.py` for on-device use.

| Stage | Operation                                                | Purpose                                          |
| ----- | -------------------------------------------------------- | ------------------------------------------------ |
| 1     | **Gaussian blur** (kernel 11×11)                         | Reduce sensor noise                              |
| 2     | **Otsu-based smart crop** (512×512, 15% padding)         | Remove background, centre hand                   |
| 3     | **Wrist removal** (distance-transform thickness profile) | Detect wrist-palm boundary and cut below it      |
| 4     | **Finger removal** (convex hull + convexity defects)     | Isolate stable metacarpal vein region            |
| 5     | **Otsu segmentation**                                    | Generate clean binary hand mask                  |
| 6     | **CLAHE** (clip limit 1.5, tile 8×8)                     | Enhance local contrast, highlight subtle veins   |
| 7     | **Sato vesselness filter** (scale 4, black ridges)       | Amplify tubular vein structures                  |
| 8     | **Feathered mask** (soft boundary fade, 12 px)           | Suppress edge artefacts from Sato filter         |
| 9     | **Tight crop + pad + resize** → 224×224                  | Fixed input size for CNN                         |
| 10    | **Z-score normalisation**                                | Standardise intensity distribution for CNN input |

The Sato vesselness filter is the dominant computational bottleneck. The pipeline outputs a single-channel (grayscale) 224×224 float32 image.

### Batch Processing (`converter.py`)

`Preprocessing/converter.py` processes an entire dataset directory in parallel using `ProcessPoolExecutor`. It mirrors the original folder structure into a `processed_dataset/` output directory and displays a live terminal progress bar. Useful for preprocessing the full training set on a workstation before uploading to Kaggle.

```bash
python converter.py
# or with explicit options:
python converter.py --dataset dataset --output processed_dataset --workers 4
```

---

## ML Model

Training script: `ml model/dorsal_palm_vein_model.py`  
Runs on Kaggle GPU.

### CNN Backbone Architecture

```
Input: 224×224×1 (grayscale NIR, Z-score normalised)

Block 1  │ Conv2D 3×3 × 32  → BN + ReLU
         │ Conv2D 3×3 × 32  → BN + ReLU
         │ MaxPool 2×2  →  112×112×32

Block 2  │ Conv2D 3×3 × 64  → BN + ReLU
         │ Conv2D 3×3 × 64  → BN + ReLU
         │ MaxPool 2×2  →   56×56×64

Block 3  │ Conv2D 3×3 × 128 → BN + ReLU
         │ Conv2D 3×3 × 128 → BN + ReLU
         │ MaxPool 2×2  →   28×28×128

Block 4  │ Conv2D 3×3 × 256 → BN + ReLU
         │ Conv2D 3×3 × 256 → BN + ReLU
         │ MaxPool 2×2  →   14×14×256

Embedding head
         │ GlobalAveragePooling2D  →  256
         │ Dense 256 + BN + ReLU
         │ Dropout 0.2
         │ Dense 128
         │ L2 Normalise  →  128-dim unit embedding
```

### Training Strategy

- **Loss**: Online semi-hard triplet loss (margin = 1.5)
- **Mining**: For each anchor — hardest positive (same identity, max distance) + semi-hard negative (different identity, farther than positive but within margin). Falls back to hardest negative when no semi-hard negative exists.
- **Batch sampler**: Identity-based (K=32 identities × N=4 images = 128 per batch)
- **Optimiser**: Adam, lr = 1×10⁻⁴ → 1×10⁻⁶ with cosine annealing (5-epoch warm-up, restarts every 20 epochs)
- **Epochs**: Up to 50 with early stopping (patience 15)
- **Embedding matching**: Euclidean distance on L2-normalised vectors, range [0, 2]

### Data Augmentation (`augmentation.py`)

| Technique                      | Parameters                                    |
| ------------------------------ | --------------------------------------------- |
| Spatial rotation               | ±12°                                          |
| Translation                    | ±7%                                           |
| Gaussian noise                 | Simulates sensor noise                        |
| Random sharpen / blur          | Simulates focus variation and motion blur     |
| Grid distortion                | Simulates skin deformation                    |
| Random rectangular patch erase | Prevents over-reliance on single vein segment |

Horizontal flipping is explicitly **excluded** to preserve left/right vein asymmetry, since left and right hands are treated as separate identities.

### Exported Artefacts

After training the script exports:

- `cnn_backbone.h5` — Keras full model
- `backbone_best.h5` — Best checkpoint
- `cnn_backbone.tflite` — Float32 TFLite model
- `cnn_backbone_ft.tflite` — Fine-tuned INT8 TFLite model (recommended for Pi deployment)
- `deployment_config.json` — Contains threshold, embedding dimension, input size
- Training curve plots

---

## On-Device Deployment (`auth_on_pi`)

### Module Overview

| File               | Role                                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `config.py`        | Single source of truth for all parameters (paths, thresholds, camera, I2C, preprocessing constants)        |
| `camera.py`        | Picamera2 singleton with fixed exposure, AWB disabled, captures 1296×972 px → grayscale                    |
| `preprocessing.py` | Full 10-stage pipeline (identical to `Preprocessing/preprocessing.py`)                                     |
| `inference.py`     | Loads TFLite model once at startup; exposes `get_embedding(raw_gray)` → 128-dim float32 array              |
| `embeddings.py`    | SQLite backend (schema: `users` + `embeddings` tables); stores 4 individual + 1 average embedding per user |
| `auth.py`          | `register()`, `verify()`, `identify()` business logic                                                      |
| `app.py`           | Flask web app with MJPEG live stream + REST endpoints                                                      |
| `cli.py`           | Terminal interface for headless use                                                                        |

### Registration Flow

1. Capture 4 images (LEFT, CENTRE, LEFT 2, CENTRE 2 hand positions)
2. Run preprocessing + inference on each → 4 individual embeddings
3. Compute L2-normalised average embedding
4. Store all 5 embeddings (4 individual + 1 average) in SQLite under the username

### Verification Flow (1:1)

1. Capture 1 image → embedding
2. Load all stored embeddings for the claimed identity
3. Compute Euclidean distance to each (average + 4 individual)
4. Use **minimum distance** (most tolerant of position variation)
5. Accept if `min_dist < THRESHOLD`

### Identification Flow (1:N)

1. Capture 1 image → embedding
2. Compare against average embeddings of every registered user
3. Return closest match if within threshold; reject otherwise

### Authentication Thresholds

| Mode            | Threshold |
| --------------- | --------- |
| Normal security | 0.30      |
| High security   | 0.25      |

The threshold can be toggled at runtime via `config.set_high_security(True/False)`.

### Flask Web App (`app.py`)

The web app serves on `0.0.0.0:5000` and is accessible from any device on the local network.

**Routes:**

| Route       | Method   | Description                            |
| ----------- | -------- | -------------------------------------- |
| `/`         | GET      | Landing page                           |
| `/stream`   | GET      | MJPEG live camera stream               |
| `/register` | GET/POST | Step-by-step registration (4 captures) |
| `/verify`   | GET/POST | 1:1 identity verification              |
| `/identify` | GET/POST | 1:N identification                     |
| `/admin`    | GET      | User management (list / delete users)  |
| `/debug`    | GET      | Debug / diagnostics page               |

### CLI (`cli.py`)

```bash
python3 cli.py
```

Provides an interactive terminal loop with commands for registration (2 captures: LEFT, CENTRE), 1:1 login, user listing, and user deletion.

---

## Installation

### Prerequisites

- Raspberry Pi 4B running Raspberry Pi OS (64-bit recommended)
- Python 3.9+
- OV5647 NoIR camera connected via CSI
- 850 nm LEDs wired to GPIO (always-on or GPIO-controlled)
- MLX90614 sensor on I2C bus 1 (SDA=GPIO2, SCL=GPIO3, 4.7 kΩ pull-ups)

### System dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev
```

### Python environment

```bash
cd auth_on_pi
python3 -m venv venv
source venv/bin/activate
pip install flask picamera2 opencv-python-headless numpy tensorflow smbus2
```

> **Note:** Install `tflite-runtime` instead of full TensorFlow if disk space is limited:
>
> ```bash
> pip install tflite-runtime
> ```
>
> Then update `inference.py` to import from `tflite_runtime.interpreter` instead of `tensorflow.lite`.

### Run the web app

```bash
cd auth_on_pi
python3 app.py
```

Access at `http://<pi-ip>:5000` from any browser on the same network. If mDNS is configured, `http://palmauth.local:5000` also works.

### Run the CLI

```bash
cd auth_on_pi
python3 cli.py
```

### Preprocessing a new dataset (workstation)

```bash
cd Preprocessing
python converter.py --dataset /path/to/raw_dataset --output /path/to/processed_dataset --workers 8
```

---

## Configuration Reference

All parameters are centralised in `auth_on_pi/config.py`.

### Paths

| Variable     | Default                      | Description       |
| ------------ | ---------------------------- | ----------------- |
| `MODEL_PATH` | `models/cnn_backbone.tflite` | TFLite model file |
| `DB_PATH`    | `data/embeddings.db`         | SQLite database   |

### Camera

| Variable        | Default       | Description                                        |
| --------------- | ------------- | -------------------------------------------------- |
| `AWB_GAINS`     | `(1.0, 1.0)`  | Fixed AWB (red, blue), disables auto white balance |
| `SHUTTER_SPEED` | `19000` µs    | Fixed exposure time                                |
| `ANALOGUE_GAIN` | `1.1`         | Sensor gain                                        |
| `CONTRAST`      | `1.6`         | Image contrast                                     |
| `CAPTURE_SIZE`  | `(1296, 972)` | Native 4:3 OV5647 resolution                       |

### Model / Matching

| Variable         | Default | Description               |
| ---------------- | ------- | ------------------------- |
| `IMG_SIZE`       | `224`   | CNN input size (px)       |
| `EMBEDDING_DIM`  | `128`   | Embedding vector length   |
| `THRESHOLD`      | `0.30`  | Normal security threshold |
| `THRESHOLD_HIGH` | `0.25`  | High security threshold   |

### Preprocessing

| Variable                  | Default     | Description                                                |
| ------------------------- | ----------- | ---------------------------------------------------------- |
| `OTSU_OFFSET`             | `-20`       | Offset applied to Otsu threshold                           |
| `WRIST_JUNCTION_RISE`     | `0.13`      | Fractional rise for wrist boundary detection               |
| `MCP_SAFETY_MARGIN`       | `0.05`      | Safety margin for MCP crop                                 |
| `MCP_DEFECT_DEPTH_MIN`    | `40` px     | Minimum convexity defect depth for finger valley detection |
| `SATO_SCALE_MIN/MAX/STEP` | `4 / 4 / 1` | Sato vesselness filter scale range                         |
| `CLAHE_CLIP`              | `1.5`       | CLAHE clip limit                                           |
| `CLAHE_TILE_SIZE`         | `8`         | CLAHE tile grid size                                       |

### Liveness (MLX90614)

| Variable        | Default   | Description                       |
| --------------- | --------- | --------------------------------- |
| `I2C_BUS`       | `1`       | Raspberry Pi I2C bus              |
| `MLX90614_ADDR` | `0x5A`    | Default I2C address               |
| `HAND_TEMP_MIN` | `30.0` °C | Minimum accepted hand temperature |
| `HAND_TEMP_MAX` | `37.0` °C | Maximum accepted hand temperature |

### Flask

| Variable | Default   | Description              |
| -------- | --------- | ------------------------ |
| `HOST`   | `0.0.0.0` | Listen on all interfaces |
| `PORT`   | `5000`    | HTTP port                |
| `DEBUG`  | `False`   | Flask debug mode         |

---

## Authors

**J Adithye¹, Abhimanyu R¹, Abhijith S Kurup¹, Adithiya S¹, Girija V R²**

¹ Undergraduate, Department of Computer Science, College of Engineering Kottarakkara, Kerala, India  
² Professor, Department of Computer Science, College of Engineering Kottarakkara, Kerala, India

---

## Citation

If you use this work, please cite:

**IJARCCE format:**

J Adithye, Abhimanyu R, Abhijith S Kurup, Adithiya S, Girija V R, "CNN-Based Dorsal Hand Vein Authentication Using Triplet Loss Metric Learning," *International Journal of Advanced Research in Computer and Communication Engineering (IJARCCE)*, DOI: 10.17148/IJARCCE.2026.15372

**BibTeX:**
```bibtex
@article{adithye2026dorsalvein,
  title   = {CNN-Based Dorsal Hand Vein Authentication Using Triplet Loss Metric Learning},
  author  = {J Adithye and Abhimanyu R and Abhijith S Kurup and Adithiya S and Girija V R},
  journal = {International Journal of Advanced Research in Computer and Communication Engineering (IJARCCE)},
  volume  = {15},
  number  = {3},
  year    = {2026},
  doi     = {10.17148/IJARCCE.2026.15372},
  url     = {https://doi.org/10.17148/IJARCCE.2026.15372}
}
```

**DOI:** [10.17148/IJARCCE.2026.15372](https://doi.org/10.17148/IJARCCE.2026.15372)

---

_Licensed under Creative Commons Attribution 4.0 International License — © IJARCCE 2026_
