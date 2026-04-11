"""Phase 2: forward hooks on the last N convolutional layers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn


class FeatureMapHook:
    """Captures Conv2d outputs R^(l) each forward."""

    def __init__(self, modules: List[nn.Module], shallow_to_deep: bool = True):
        self.modules = modules
        self.shallow_to_deep = shallow_to_deep
        self._handles: List[Any] = []
        self._storage: Dict[int, torch.Tensor] = {}
        self._order: List[int] = list(range(len(modules)))

    def register(self) -> None:
        self.clear()

        def make_hook(idx: int) -> Callable:
            def hook(_m: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
                self._storage[idx] = out.detach()

            return hook

        for i, m in enumerate(self.modules):
            h = m.register_forward_hook(make_hook(i))
            self._handles.append(h)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self._storage.clear()

    def get_maps(self) -> List[torch.Tensor]:
        """Return feature maps in shallow-to-deep order."""
        order = self._order if self.shallow_to_deep else list(reversed(self._order))
        return [self._storage[i] for i in order]

    def __enter__(self) -> "FeatureMapHook":
        self.register()
        return self

    def __exit__(self, *args) -> None:
        self.remove()


def attach_hooks_to_convs(conv_modules: List[nn.Module]) -> FeatureMapHook:
    return FeatureMapHook(conv_modules, shallow_to_deep=True)
