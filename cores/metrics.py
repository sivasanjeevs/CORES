"""Phase 4: Response magnitude (RM) and response frequency (RF)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _spatial_max_min(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """R: (B, C, H, W) -> max, min per map: (B, C), (B, C)."""
    mx = R.amax(dim=(2, 3))
    mn = R.amin(dim=(2, 3))
    return mx, mn


def response_magnitude_positive(R: torch.Tensor, tau_plus: torch.Tensor) -> torch.Tensor:
    """RM+ — mean over maps of relu(max(R_i) - tau+). R: (B, C, H, W)."""
    mx, _ = _spatial_max_min(R)
    return F.relu(mx - tau_plus).mean(dim=1)


def response_magnitude_negative(R: torch.Tensor, tau_minus: torch.Tensor) -> torch.Tensor:
    """RM- — mean over maps of relu(tau- - min(R_i))."""
    _, mn = _spatial_max_min(R)
    return F.relu(tau_minus - mn).mean(dim=1)


def response_frequency_positive(R: torch.Tensor, tau_plus: torch.Tensor) -> torch.Tensor:
    """RF+ — mean over maps of I(max(R_i) > tau+)."""
    mx, _ = _spatial_max_min(R)
    return (mx > tau_plus).float().mean(dim=1)


def response_frequency_negative(R: torch.Tensor, tau_minus: torch.Tensor) -> torch.Tensor:
    """RF- — mean over maps of I(min(R_i) < tau-)."""
    _, mn = _spatial_max_min(R)
    return (mn < tau_minus).float().mean(dim=1)


def cores_layer_metrics(
    R_bar: torch.Tensor,
    R_under: torch.Tensor,
    tau_plus: torch.Tensor,
    tau_minus: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns RM+, RM-, RF+, RF- each shape (B,)."""
    rmp = response_magnitude_positive(R_bar, tau_plus)
    rmn = response_magnitude_negative(R_under, tau_minus)
    rfp = response_frequency_positive(R_bar, tau_plus)
    rfn = response_frequency_negative(R_under, tau_minus)
    return rmp, rmn, rfp, rfn
