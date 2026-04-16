"""End-to-end CORES: hooks, backtracking, calibration, scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backtrack import backtrack_kernel_indices, extract_fc_weight
from .calibration import calibrate_per_layer, synthetic_noise_batch
from .feature_hooks import FeatureMapHook, attach_hooks_to_convs
from .models_cifar import get_last_conv_modules, get_stage_boundary_convs
from .scoring import aggregate_layers, anomaly_score_from_cores, layer_score


def gather_channel_maps(R: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """R: (B,C,H,W), idx: (K,) -> (B,K,H,W)."""
    return R.index_select(1, idx.to(R.device))


@dataclass
class CalibrationState:
    tau_plus: List[torch.Tensor]
    tau_minus: List[torch.Tensor]


class CoresPipeline:
    def __init__(self, model: nn.Module):
        self.model = model
        self.conv_modules = get_last_conv_modules(model)
        self.stage_convs = get_stage_boundary_convs(model)
        self.fc_weight = extract_fc_weight(model)
        self._hook: Optional[FeatureMapHook] = None
        self.calib: Optional[CalibrationState] = None

    def _ensure_hook(self) -> FeatureMapHook:
        if self._hook is None:
            self._hook = attach_hooks_to_convs(self.conv_modules)
            self._hook.register()
        return self._hook

    def remove_hooks(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def forward_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self._ensure_hook()
        h.clear()
        logits = self.model(x)
        return logits, h.get_maps()

    def calibrate(
        self,
        device: torch.device,
        shape: Tuple[int, int, int] = (3, 32, 32),
        noise_batches: int = 4,
        batch_size: int = 64,
        target_fpr: float = 0.05,
    ) -> CalibrationState:
        self.model.eval()
        h = self._ensure_hook()

        def maps_fn() -> List[torch.Tensor]:
            return h.get_maps()

        mx_cat: List[torch.Tensor] = []
        mn_cat: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(noise_batches):
                for mode in ("gaussian", "uniform"):
                    x = synthetic_noise_batch(batch_size, shape, device, mode)
                    h.clear()
                    _ = self.model(x)
                    maps = h.get_maps()
                    if not mx_cat:
                        mx_cat = [[] for _ in maps]
                        mn_cat = [[] for _ in maps]
                    for li, R in enumerate(maps):
                        mx_cat[li].append(R.amax(dim=(2, 3)).cpu())
                        mn_cat[li].append(R.amin(dim=(2, 3)).cpu())
        mx_layers = [torch.cat(parts, dim=0) for parts in mx_cat]
        mn_layers = [torch.cat(parts, dim=0) for parts in mn_cat]
        taup, taun = calibrate_per_layer(mx_layers, mn_layers, target_fpr=target_fpr)
        self.calib = CalibrationState(
            tau_plus=[t.to(device) for t in taup],
            tau_minus=[t.to(device) for t in taun],
        )
        return self.calib

    def scores_single_forward(
        self,
        x: torch.Tensor,
        fraction: float = 0.2,
        lambda1: float = 10.0,
        lambda2: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (1,3,32,32). Returns (cores_score, ood_score) scalars for the sample.
        """
        if self.calib is None:
            raise RuntimeError("Call calibrate() first.")
        logits, maps = self.forward_maps(x)
        probs = F.softmax(logits, dim=1)[0]
        Fw = self.fc_weight
        bar_list, under_list = backtrack_kernel_indices(Fw, probs, self.stage_convs, fraction)

        layer_S: List[torch.Tensor] = []
        for li, R in enumerate(maps):
            tb = self.calib.tau_plus[li]
            tn = self.calib.tau_minus[li]
            Rb = gather_channel_maps(R, bar_list[li])
            Ru = gather_channel_maps(R, under_list[li])
            S = layer_score(Rb, Ru, tb, tn, lambda1=lambda1, lambda2=lambda2)
            layer_S.append(S)
        S_total = aggregate_layers(layer_S)
        ood = anomaly_score_from_cores(S_total)
        return S_total.view(()), ood.view(())

    def scores_batch_same_indices(
        self,
        x: torch.Tensor,
        bar_list: Sequence[torch.Tensor],
        under_list: Sequence[torch.Tensor],
        lambda1: float = 10.0,
        lambda2: float = 1.0,
    ) -> torch.Tensor:
        """When all samples share the same kernel mask (approximation for speed)."""
        if self.calib is None:
            raise RuntimeError("Call calibrate() first.")
        logits, maps = self.forward_maps(x)
        layer_S: List[torch.Tensor] = []
        for li, R in enumerate(maps):
            tb = self.calib.tau_plus[li]
            tn = self.calib.tau_minus[li]
            Rb = gather_channel_maps(R, bar_list[li])
            Ru = gather_channel_maps(R, under_list[li])
            S = layer_score(Rb, Ru, tb, tn, lambda1=lambda1, lambda2=lambda2)
            layer_S.append(S)
        return aggregate_layers(layer_S)
