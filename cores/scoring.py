"""Phase 6: layer-wise CORES score aggregation."""

from __future__ import annotations

from typing import List, Sequence

import torch

from .metrics import cores_layer_metrics


def layer_score(
    R_bar: torch.Tensor,
    R_under: torch.Tensor,
    tau_plus: torch.Tensor,
    tau_minus: torch.Tensor,
    lambda1: float = 10.0,
    lambda2: float = 1.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Calculated in Log-Space to prevent float32 underflow from power of 10.
    log(S) = λ1*log(RM+) + λ1*log(RM-) + λ2*log(RF+) + λ2*log(RF-)
    """
    rmp, rmn, rfp, rfn = cores_layer_metrics(R_bar, R_under, tau_plus, tau_minus)
    log_S = (
        lambda1 * torch.log(rmp + eps)
        + lambda1 * torch.log(rmn + eps)
        + lambda2 * torch.log(rfp + eps)
        + lambda2 * torch.log(rfn + eps)
    )
    return log_S


def aggregate_layers(
    layer_scores: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean across layers — shape (B,)."""
    return torch.stack(list(layer_scores), dim=0).mean(dim=0)


def anomaly_score_from_cores(log_S: torch.Tensor) -> torch.Tensor:
    """
    Since we are in log space, a higher log_S means it is MORE in-distribution.
    We return negative log_S so that larger values = more anomalous (OOD).
    """
    return -log_S
