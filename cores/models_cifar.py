"""CIFAR-style ResNet-18 / WideResNet-28-10 with 32×32 inputs."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet for 32×32 images (modified stem: 3×3 conv, stride 1, no maxpool)."""

    def __init__(self, block, num_blocks: List[int], num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet18_cifar(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# --- WideResNet-28-10 (Zagoruyko & Komodakis style) ---


class WideBasic(nn.Module):
    def __init__(self, in_planes: int, planes: int, dropout: float, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNetCIFAR(nn.Module):
    def __init__(self, depth: int, widen_factor: int, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        n_channels = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.layer1 = self._make_layer(WideBasic, n_channels[0], n_channels[1], n, dropout, stride=1)
        self.layer2 = self._make_layer(WideBasic, n_channels[1], n_channels[2], n, dropout, stride=2)
        self.layer3 = self._make_layer(WideBasic, n_channels[2], n_channels[3], n, dropout, stride=2)
        self.bn_out = nn.BatchNorm2d(n_channels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)

    def _make_layer(self, block, in_planes: int, planes: int, num_blocks: int, dropout: float, stride: int):
        layers = [block(in_planes, planes, dropout, stride)]
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes, dropout, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_out(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def wideresnet_28_10(num_classes: int = 10, dropout: float = 0.0) -> WideResNetCIFAR:
    return WideResNetCIFAR(depth=28, widen_factor=10, num_classes=num_classes, dropout=dropout)


def get_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return resnet18_cifar(num_classes=num_classes)
    if name in ("wideresnet_28_10", "wrn_28_10", "wrn28_10"):
        return wideresnet_28_10(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {name}")


def last_conv_modules_resnet(model: ResNetCIFAR) -> List[nn.Conv2d]:
    """Last conv2d of each of the four residual stages (paper-style hooks)."""
    return [
        model.layer1[-1].conv2,
        model.layer2[-1].conv2,
        model.layer3[-1].conv2,
        model.layer4[-1].conv2,
    ]


def last_conv_modules_wrn(model: WideResNetCIFAR) -> List[nn.Conv2d]:
    """Four conv layers: stem + last conv in each wide group (CIFAR-style depth)."""
    return [
        model.conv1,
        model.layer1[-1].conv2,
        model.layer2[-1].conv2,
        model.layer3[-1].conv2,
    ]


def get_last_conv_modules(model: nn.Module) -> List[nn.Conv2d]:
    if isinstance(model, ResNetCIFAR):
        return last_conv_modules_resnet(model)
    if isinstance(model, WideResNetCIFAR):
        return last_conv_modules_wrn(model)
    raise TypeError(f"Unsupported model type for hook enumeration: {type(model)}")


def stage_boundary_convs_resnet(model: ResNetCIFAR) -> List[nn.Conv2d]:
    """Conv layers mapping stage outputs shallow -> deep (first conv of layer2/3/4)."""
    return [
        model.layer2[0].conv1,
        model.layer3[0].conv1,
        model.layer4[0].conv1,
    ]


def stage_boundary_convs_wrn(model: WideResNetCIFAR) -> List[nn.Conv2d]:
    """Stage transitions for WRN (16 -> 160 -> 320 -> 640 channels)."""
    return [
        model.layer1[0].conv1,
        model.layer2[0].conv1,
        model.layer3[0].conv1,
    ]


def get_stage_boundary_convs(model: nn.Module) -> List[nn.Conv2d]:
    if isinstance(model, ResNetCIFAR):
        return stage_boundary_convs_resnet(model)
    if isinstance(model, WideResNetCIFAR):
        return stage_boundary_convs_wrn(model)
    raise TypeError(type(model))
