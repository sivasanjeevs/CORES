"""Default hyperparameters and paths."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CoresConfig:
    # Data (32×32 for CIFAR-scale experiments)
    data_root: Path = Path("./data")
    id_dataset: str = "cifar10"  # cifar10 | cifar100
    ood_dataset: str = "svhn"  # svhn | textures | lsun_resize

    # Model
    architecture: str = "resnet18"  # resnet18 | wideresnet_28_10
    num_classes: int = 10
    pretrained: bool = False  # torchvision ImageNet weights (stem adapted); train for CIFAR OOD

    # Hooks: last N conv layers (paper: 4 for CIFAR, 5 for ImageNet-scale)
    num_hook_layers: int = 4

    # Kernel selection
    kernel_top_fraction: float = 0.2

    # Scoring (Phase 6)
    lambda1: float = 10.0
    lambda2: float = 1.0

    # Calibration
    noise_batch_size: int = 64
    noise_num_batches: int = 4
    calib_lr: float = 0.05
    calib_steps: int = 50

    # Eval
    batch_size: int = 128
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 0

    # Checkpoint (optional)
    checkpoint: Optional[str] = None
