"""In-distribution and OOD dataloaders (32×32 CIFAR-scale)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def cifar_transform_test() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def get_id_dataloader(
    name: str,
    root: Path,
    batch_size: int,
    num_workers: int = 2,
    download: bool = True,
    subset: Optional[int] = None,
) -> DataLoader:
    root = Path(root)

    if name.lower() != "cifar10":
        raise ValueError(f"Unsupported ID dataset: {name}")

    ds = datasets.CIFAR10(
        root,
        train=False,
        download=download,
        transform=cifar_transform_test(),
    )

    if subset is not None and subset < len(ds):
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[:subset].tolist()
        ds = Subset(ds, idx)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
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
            transform=cifar_transform_test(),
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
                transform=cifar_transform_test(),
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
    return 10
