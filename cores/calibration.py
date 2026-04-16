"""Phase 5: synthetic noise and threshold calibration."""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch


def synthetic_noise_batch(
    batch_size: int,
    shape: Tuple[int, int, int],
    device: torch.device,
    mode: str = "gaussian",
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
) -> torch.Tensor:
    """Gaussian or uniform noise, then CIFAR-style normalization (matches ID pipeline)."""
    c, h, w = shape
    if mode == "gaussian":
        # Generate noise in standard image space [0, 1], then apply CIFAR normalization.
        x = torch.randn(batch_size, c, h, w, device=device) * 0.5 + 0.5
        x = torch.clamp(x, 0.0, 1.0)
    elif mode == "uniform":
        # Uniform noise directly in image bounds [0, 1].
        x = torch.rand(batch_size, c, h, w, device=device)
    else:
        raise ValueError(mode)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


def noise_feature_max_min(
    model: torch.nn.Module,
    hook_maps_fn: Callable[[], List[torch.Tensor]],
    noise_batches: int,
    batch_size: int,
    shape: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[list, list]:
    """
    Run noise through the network and collect per-layer spatial max/min over channels
    for hooked feature maps. Returns lists of tensors (N, C) for max and min per layer.
    """
    model.eval()
    mx_all: list = []
    mn_all: list = []
    first = True
    with torch.no_grad():
        for _ in range(noise_batches):
            g = synthetic_noise_batch(batch_size, shape, device, "gaussian")
            u = synthetic_noise_batch(batch_size, shape, device, "uniform")
            for x in (g, u):
                _ = model(x)
                maps = hook_maps_fn()
                if first:
                    mx_all = [[] for _ in maps]
                    mn_all = [[] for _ in maps]
                    first = False
                for li, R in enumerate(maps):
                    mx = R.amax(dim=(2, 3))
                    mn = R.amin(dim=(2, 3))
                    mx_all[li].append(mx.cpu())
                    mn_all[li].append(mn.cpu())
    mx_cat = [torch.cat(parts, dim=0) for parts in mx_all]
    mn_cat = [torch.cat(parts, dim=0) for parts in mn_all]
    return mx_cat, mn_cat


def calibrate_thresholds_min_fpr(
    mx_layers: list,
    mn_layers: list,
    target_fpr: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per layer choose tau+ as high quantile of max responses on noise (few max > tau+),
    tau- as low quantile of min responses (few min < tau-).
    Scalar per layer returned as tensors length L — here we use **one global** tau+, tau-
    by pooling all layers' statistics (simple); alternatively stack per layer.
    """
    device = mx_layers[0].device if isinstance(mx_layers[0], torch.Tensor) else torch.device("cpu")
    mx_flat = torch.cat([m.flatten() for m in mx_layers])
    mn_flat = torch.cat([m.flatten() for m in mn_layers])
    # Quantile such that fraction of values above tau+ is ~target_fpr for max side
    q_hi = 1.0 - target_fpr
    q_lo = target_fpr
    tau_plus = torch.quantile(mx_flat.to(torch.float32), q_hi).to(device)
    tau_minus = torch.quantile(mn_flat.to(torch.float32), q_lo).to(device)
    return tau_plus, tau_minus


def calibrate_per_layer(
    mx_layers: list,
    mn_layers: list,
    target_fpr: float = 0.01,
) -> Tuple[list, list]:
    """Return lists of scalar tensors tau+_l, tau-_l per hooked layer."""
    taup, taun = [], []
    for mx, mn in zip(mx_layers, mn_layers):
        mx_flat = mx.flatten().float()
        mn_flat = mn.flatten().float()
        q_hi = 1.0 - target_fpr
        q_lo = target_fpr
        taup.append(torch.quantile(mx_flat, q_hi))
        taun.append(torch.quantile(mn_flat, q_lo))
    return taup, taun
