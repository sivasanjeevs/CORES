"""Phase 3: FC weights and recursive (stage-wise) kernel selection."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def aggregate_conv_influence(conv: nn.Conv2d) -> torch.Tensor:
    """Shape (out_channels, in_channels)."""
    w = conv.weight.data.abs()
    return w.sum(dim=(2, 3))


def extract_fc_weight(model: nn.Module) -> torch.Tensor:
    """Return F with shape (num_classes, dim) for nn.Linear."""
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError("Model must have an nn.Linear `fc` layer.")
    return model.fc.weight.data.detach()


def top_fraction_indices(values: torch.Tensor, fraction: float, largest: bool = True) -> torch.Tensor:
    """1D tensor values -> top `fraction` of indices (at least 1)."""
    n = values.numel()
    k = max(1, int(round(n * fraction)))
    k = min(k, n)
    _, idx = torch.topk(values, k, largest=largest)
    return idx


def select_last_layer_indices(
    F_weight: torch.Tensor,
    probs: torch.Tensor,
    fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    F: (num_classes, C). probs: (num_classes,) for one sample.
    Returns bar_I (top fraction of F[c_max]), under_I (bottom fraction of F[c_min]).
    """
    c_max = int(torch.argmax(probs).item())
    c_min = int(torch.argmin(probs).item())
    row_max = F_weight[c_max]
    row_min = F_weight[c_min]
    bar_I = top_fraction_indices(row_max, fraction, largest=True)
    under_I = top_fraction_indices(row_min, fraction, largest=False)
    return bar_I, under_I, c_max, c_min


def propagate_to_prev_stage(
    W: torch.Tensor,
    child_indices: torch.Tensor,
    fraction: float,
    bar: bool,
) -> torch.Tensor:
    """
    W: (out_ch, in_ch) aggregated influence from stage below -> above.
    child_indices: subset of rows (output channels at deeper stage).
    """
    device = W.device
    mask = torch.zeros(W.shape[0], device=device)
    mask[child_indices] = 1.0
    scores = (W * mask.unsqueeze(1)).sum(dim=0)
    return top_fraction_indices(scores, fraction, largest=bar)


def backtrack_kernel_indices(
    fc_weight: torch.Tensor,
    probs: torch.Tensor,
    stage_boundary_convs: Sequence[nn.Conv2d],
    fraction: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Returns lists `bar` and `under` of length = 1 + len(stage_boundary_convs), shallow -> deep,
    matching hooked conv layers (4 for ResNet when three stage matrices are used + FC stage).
    """
    device = fc_weight.device
    bar_deep, under_deep, _, _ = select_last_layer_indices(fc_weight, probs, fraction)

    mats = [aggregate_conv_influence(c).to(device) for c in stage_boundary_convs]
    mats_rev = list(reversed(mats))

    bar_seq: List[torch.Tensor] = []
    under_seq: List[torch.Tensor] = []
    cur_b, cur_u = bar_deep.clone(), under_deep.clone()
    for W in mats_rev:
        bar_seq.append(cur_b)
        under_seq.append(cur_u)
        cur_b = propagate_to_prev_stage(W, cur_b, fraction, bar=True)
        cur_u = propagate_to_prev_stage(W, cur_u, fraction, bar=False)
    bar_seq.append(cur_b)
    under_seq.append(cur_u)
    bar_seq.reverse()
    under_seq.reverse()
    return bar_seq, under_seq
