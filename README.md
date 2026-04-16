# CORES: Convolutional Response-based Score for OOD Detection

This repository contains an implementation of the research paper **"CORES: Convolutional Response-based Score for Out-of-distribution Detection"**.

CORES is a post-hoc OOD scoring method that uses internal convolutional responses to assign an anomaly score, then evaluates the separation between ID and OOD using `AUROC` and `FPR95`.


## Prerequisites

- Python 3.10+ (tested with Python 3.12)
- Linux/macOS (or WSL for Windows)
- Optional: NVIDIA GPU with CUDA support (for `--device cuda`)

## Installation (CPU & GPU)

### Option 1: CPU-only setup (provided script)

From the repo root:

```bash
cd /home/sivasanjeev/dm
bash setup_venv.sh
source .venv/bin/activate
```

`setup_venv.sh` creates a virtual environment and installs:
- CPU PyTorch wheel (`--index-url https://download.pytorch.org/whl/cpu`)
- dependencies from `requirements.txt`

### Option 2: GPU (CUDA) setup

Create a venv and install a CUDA-enabled PyTorch wheel that matches your CUDA version.
Example (CUDA 12.1):

```bash
cd /home/sivasanjeev/dm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If your CUDA version is different, replace `cu121` with the correct wheel tag.

## Quick Start

### 1) Download the pretrained checkpoint

This repo uses a pretrained CIFAR-10 ResNet-18 checkpoint for the paper setting.
You can generate/download it with:

```bash
python3 model.py
```

This creates `resnet18_cifar10_ready.pth` in the repo root.

### 2) Run the evaluation

For the recommended paper-style setting (works with `--checkpoint auto`):

```bash
python3 main.py --device cpu --id cifar10 --arch resnet18 --ood svhn --train-epochs 0 --checkpoint auto
```

## Usage Guide

### Running on CPU

```bash
cd /home/sivasanjeev/dm
source .venv/bin/activate

python3 main.py \
  --device cpu \
  --id cifar10 \
  --arch resnet18 \
  --ood svhn \
  --train-epochs 0 \
  --checkpoint auto
```

### Running on GPU

```bash
cd /home/sivasanjeev/dm
source .venv/bin/activate

python3 main.py \
  --device cuda \
  --id cifar10 \
  --arch resnet18 \
  --ood svhn \
  --train-epochs 0 \
  --checkpoint auto
```

### Multiple OOD datasets in one run

```bash
python3 main.py \
  --device cuda \
  --id cifar10 \
  --arch resnet18 \
  --ood svhn textures lsun_resize \
  --train-epochs 0 \
  --checkpoint auto
```

Notes on OOD data:
- `svhn` downloads automatically via torchvision
- `textures` uses the DTD test split
- `lsun_resize` uses torchvision's LSUN dataset and may require manual data under:
  - `./data/lsun/bedroom_val/`

### Hyperparameter Tuning

Key CLI flags (defaults come from `main.py`):
- `--fraction` (default `0.2`): fraction of kernels used during recursive kernel tracing
- `--calib-batches` (default `20`): number of synthetic noise batches used to calibrate thresholds
- `--target-fpr` (default `0.05`): quantile target used when computing `tau_plus` / `tau_minus`

Example:

```bash
python3 main.py \
  --device cuda \
  --id cifar10 \
  --arch resnet18 \
  --ood svhn \
  --train-epochs 0 \
  --checkpoint auto \
  --calib-batches 20 \
  --target-fpr 0.05
```

For a fast smoke test (much quicker than full evaluation), limit evaluation batches:

```bash
python3 main.py \
  --device cpu \
  --id cifar10 \
  --arch resnet18 \
  --ood svhn \
  --train-epochs 0 \
  --checkpoint auto \
  --eval-batch-cap 2
```

### Checkpoint behavior (`--checkpoint auto` vs file path)

- `--checkpoint auto` loads the pretrained ResNet-18 CIFAR-10 weights, but only for:
  - `--id cifar10 --arch resnet18`
- `--checkpoint ./resnet18_cifar10_ready.pth` loads the same checkpoint from a local file.
- `--checkpoint none` skips loading and runs with randomly initialized weights (not recommended for paper results).

## Project Structure

Typical layout:

```text
.
├── main.py                 # end-to-end calibrate + evaluate
├── model.py                # downloads/converts pretrained checkpoint
├── setup_venv.sh           # creates venv + installs CPU-only PyTorch
├── requirements.txt
├── resnet18_cifar10_ready.pth
└── cores/
    ├── pipeline.py         # hooks + calibration + scoring
    ├── scoring.py          # CORES scoring (log-space scoring)
    ├── calibration.py      # synthetic noise + threshold calibration
    ├── backtrack.py        # Equation 6/8 kernel selection
    ├── data_loaders.py     # CIFAR / SVHN / textures / LSUN loaders
    ├── eval_metrics.py     # AUROC + FPR95
    └── models_cifar.py     # ResNet-18 and WideResNet implementations
```

