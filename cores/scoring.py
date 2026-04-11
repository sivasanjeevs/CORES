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
    S(R̄ ∪ R̲) = RM+(R̄)^λ1 · RM-(R̲)^λ1 · RF+(R̄)^λ2 · RF-(R̲)^λ2
    Shape (B,).
    """
    rmp, rmn, rfp, rfn = cores_layer_metrics(R_bar, R_under, tau_plus, tau_minus)
    # Large λ1 needs a floor before pow to avoid all-zero products on random weights.
    return (
        (rmp + eps).pow(lambda1)
        * (rmn + eps).pow(lambda1)
        * (rfp + eps).pow(lambda2)
        * (rfn + eps).pow(lambda2)
    )


def aggregate_layers(
    layer_scores: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean across layers — shape (B,)."""
    return torch.stack(list(layer_scores), dim=0).mean(dim=0)


def anomaly_score_from_cores(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert CORES score S (larger often = more ID-like) into an OOD score where **larger = OOD**.
    """
    return 1.0 / (S + eps)
