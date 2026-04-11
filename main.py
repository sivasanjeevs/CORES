#!/usr/bin/env python3
"""CORES OOD detection — train (optional), calibrate, evaluate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from cores.data_loaders import dataset_num_classes, get_id_dataloader, get_ood_dataloader
from cores.eval_metrics import auroc_fpr95
from cores.models_cifar import get_model
from cores.pipeline import CoresPipeline


def train_classifier(
    model: nn.Module,
    train_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 0.1,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        total, correct = 0, 0
        for x, y in tqdm(train_loader, desc=f"train ep{ep+1}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        sched.step()
        print(f"  epoch {ep+1}: acc={correct/total:.4f}")


def collect_scores(
    pipeline: CoresPipeline,
    loader,
    device: torch.device,
    label: int,
    fraction: float,
    max_batches: int | None = None,
) -> list[float]:
    pipeline.model.eval()
    out: list[float] = []
    for bi, (x, _) in enumerate(tqdm(loader, desc=f"score label={label}")):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        for i in range(x.size(0)):
            _, ood = pipeline.scores_single_forward(x[i : i + 1], fraction=fraction)
            out.append(float(ood.detach().cpu()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="CORES OOD detection")
    p.add_argument("--data-root", type=Path, default=Path("./data"))
    p.add_argument("--id", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--ood", default="svhn", choices=["svhn", "textures", "lsun_resize"])
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "wideresnet_28_10"])
    p.add_argument("--train-epochs", type=int, default=0, help="Fine-tune classifier on ID data (0=skip)")
    p.add_argument("--calib-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-cap", type=int, default=None, help="Limit batches per split for quick runs")
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--target-fpr", type=float, default=0.01, help="Noise calibration quantile target")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    num_classes = dataset_num_classes(args.id)
    model = get_model(args.arch, num_classes=num_classes).to(device)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd, strict=True)

    train_loader = get_id_dataloader(
        args.id, args.data_root, train=True, batch_size=args.batch_size, download=True
    )
    test_id = get_id_dataloader(
        args.id, args.data_root, train=False, batch_size=args.batch_size, download=True
    )
    try:
        test_ood = get_ood_dataloader(
            args.ood, args.data_root, batch_size=args.batch_size, download=True
        )
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if args.train_epochs > 0:
        train_classifier(model, train_loader, device, epochs=args.train_epochs)

    pipe = CoresPipeline(model)
    print("Calibrating thresholds on synthetic noise...")
    pipe.calibrate(
        device=device,
        shape=(3, 32, 32),
        noise_batches=args.calib_batches,
        batch_size=args.batch_size,
        target_fpr=args.target_fpr,
    )

    print("Scoring ID...")
    id_scores = collect_scores(
        pipe, test_id, device, label=0, fraction=args.fraction, max_batches=args.eval_batch_cap
    )
    print("Scoring OOD...")
    ood_scores = collect_scores(
        pipe, test_ood, device, label=1, fraction=args.fraction, max_batches=args.eval_batch_cap
    )

    y = np.array([0] * len(id_scores) + [1] * len(ood_scores))
    s = np.array(id_scores + ood_scores)
    auroc, fpr95 = auroc_fpr95(y, s)
    print(f"Samples: ID={len(id_scores)}, OOD={len(ood_scores)}")
    print(f"AUROC: {auroc:.4f}")
    print(f"FPR95: {fpr95:.4f}")
    pipe.remove_hooks()


if __name__ == "__main__":
    main()
