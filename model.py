from __future__ import annotations

import urllib.request
from pathlib import Path

import torch


HF_URL = "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin"


def download_resnet18_cifar10_ready(dst_path: Path | str = "resnet18_cifar10_ready.pth") -> Path:
    """
    Download pretrained CIFAR-10 ResNet-18 weights and convert the state_dict keys
    to match this repo's CIFAR ResNet implementation (strict load).
    """
    dst_path = Path(dst_path)
    if dst_path.exists():
        return dst_path

    raw_path = dst_path.with_suffix(".raw.pth")
    print("Downloading CIFAR-10 ResNet-18 weights from Hugging Face...")
    urllib.request.urlretrieve(HF_URL, raw_path.as_posix())

    print("Cleaning dictionary keys to match your custom architecture...")
    state_dict = torch.load(raw_path, map_location="cpu")

    # Remove extra prefixes like 'model.' or 'module.' that cause strict=True loading errors.
    # Also normalize torchvision ResNet naming:
    #   torchvision BasicBlock uses `downsample.*`
    #   this repo's BasicBlock uses `shortcut.*`
    clean_dict = {
        k.replace("model.", "")
        .replace("module.", "")
        .replace("downsample.", "shortcut."): v
        for k, v in state_dict.items()
    }

    torch.save(clean_dict, dst_path)

    # Best-effort cleanup of intermediate raw file.
    try:
        raw_path.unlink()
    except OSError:
        pass

    print(f"Done! Checkpoint saved as {dst_path.as_posix()}.")
    return dst_path


if __name__ == "__main__":
    download_resnet18_cifar10_ready()