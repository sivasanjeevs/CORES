"""In-distribution and OOD dataloaders (32×32 CIFAR-scale)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def cifar_transform_train() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def cifar_transform_test() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def svhn_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def imagenet_norm_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def get_id_dataloader(
    name: str,
    root: Path,
    train: bool,
    batch_size: int,
    num_workers: int = 2,
    download: bool = True,
    subset: Optional[int] = None,
) -> DataLoader:
    name = name.lower()
    root = Path(root)
    if name.startswith("imagenet"):
        raise NotImplementedError(
            "ImageNet-1k ID data is not wired in this repo. Use CIFAR-10/100 for 32×32 experiments "
            "or add an ImageFolder-based loader and set num_classes=1000."
        )

    if name == "cifar10":
        ds = datasets.CIFAR10(
            root,
            train=train,
            download=download,
            transform=cifar_transform_train() if train else cifar_transform_test(),
        )
    elif name == "cifar100":
        ds = datasets.CIFAR100(
            root,
            train=train,
            download=download,
            transform=cifar_transform_train() if train else cifar_transform_test(),
        )
    else:
        raise ValueError(f"Unsupported ID dataset: {name}")

    if subset is not None and subset < len(ds):
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[:subset].tolist()
        ds = Subset(ds, idx)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_ood_dataloader(
    name: str,
    root: Path,
    batch_size: int,
    num_workers: int = 2,
    download: bool = True,
    subset: Optional[int] = None,
) -> DataLoader:
    """OOD sets resized/normalized to match CIFAR preprocessing when noted."""
    name = name.lower()
    root = Path(root)

    if name == "svhn":
        ds = datasets.SVHN(
            root / "svhn",
            split="test",
            download=download,
            transform=svhn_transform(),
        )
    elif name in ("textures", "dtd"):
        ds = datasets.DTD(
            root / "dtd",
            split="test",
            download=download,
            transform=cifar_transform_test(),
        )
    elif name in ("lsun", "lsun_resize", "lsun-r"):
        try:
            ds = datasets.LSUN(
                root / "lsun",
                classes="bedroom_val",
                transform=imagenet_norm_transform(),
            )
        except Exception:
            raise RuntimeError(
                "LSUN requires manual download or a local folder; use --ood svhn or textures, "
                "or place LSUN bedroom_val under data/lsun."
            ) from None
    else:
        raise ValueError(f"Unsupported OOD dataset: {name}")

    if subset is not None and subset < len(ds):
        g = torch.Generator().manual_seed(1)
        idx = torch.randperm(len(ds), generator=g)[:subset].tolist()
        ds = Subset(ds, idx)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def dataset_num_classes(id_name: str) -> int:
    if id_name.lower() == "cifar100":
        return 100
    return 10
