#!/usr/bin/env python3
"""CORES OOD detection — train (optional), calibrate, evaluate."""

from __future__ import annotations

import argparse
import random
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
from model import download_resnet18_cifar10_ready


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU-only runs still benefit from deterministic settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    p.add_argument("--ood", default=["svhn"], nargs="+", choices=["svhn", "textures", "lsun_resize"])
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "wideresnet_28_10"])
    p.add_argument("--train-epochs", type=int, default=0, help="Fine-tune classifier on ID data (0=skip)")
    p.add_argument("--calib-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-cap", type=int, default=None, help="Limit batches per split for quick runs")
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--target-fpr", type=float, default=0.05, help="Noise calibration quantile target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="auto",
        help="Pretrained checkpoint path, or 'auto' to download CIFAR-10 ResNet-18, or 'none' to skip loading.",
    )
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    num_classes = dataset_num_classes(args.id)
    model = get_model(args.arch, num_classes=num_classes).to(device)

    checkpoint_mode = (args.checkpoint or "auto").lower()
    if checkpoint_mode != "none":
        if checkpoint_mode == "auto":
            if args.id != "cifar10" or args.arch != "resnet18":
                raise ValueError(
                    "Checkpoint auto-download is only wired for the paper setting: "
                    "`--id cifar10` + `--arch resnet18`. "
                    "For other combinations, pass an explicit `--checkpoint`."
                )
            ckpt_path = Path(__file__).resolve().parent / "resnet18_cifar10_ready.pth"
            if not ckpt_path.exists():
                # Download + key conversion happens only when the file is missing.
                try:
                    ckpt_path = download_resnet18_cifar10_ready(ckpt_path)
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(
                        f"Failed to auto-download checkpoint. Missing: {ckpt_path}\n"
                        f"Either fix your network or run `python model.py` once to create it, then re-run with:\n"
                        f"  --checkpoint {ckpt_path.as_posix()}"
                    ) from e
        else:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"--checkpoint not found: {ckpt_path.as_posix()}")

        sd = torch.load(ckpt_path.as_posix(), map_location=device)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded pretrained checkpoint: {ckpt_path.as_posix()}")

    test_id = get_id_dataloader(
        args.id, args.data_root, train=False, batch_size=args.batch_size, download=True
    )

    if args.train_epochs > 0:
        train_loader = get_id_dataloader(
            args.id, args.data_root, train=True, batch_size=args.batch_size, download=True
        )
    else:
        train_loader = None

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

    print(
        f"CORES params: fraction={args.fraction}, target_fpr={args.target_fpr}, "
        f"calib_batches={args.calib_batches}"
    )
    results: list[tuple[str, float, float]] = []
    for ood_name in args.ood:
        try:
            test_ood = get_ood_dataloader(
                ood_name, args.data_root, batch_size=args.batch_size, download=True
            )
        except RuntimeError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

        print(f"Scoring OOD ({ood_name})...")
        ood_scores = collect_scores(
            pipe,
            test_ood,
            device,
            label=1,
            fraction=args.fraction,
            max_batches=args.eval_batch_cap,
        )

        y = np.array([0] * len(id_scores) + [1] * len(ood_scores))
        s = np.array(id_scores + ood_scores)
        auroc, fpr95 = auroc_fpr95(y, s)
        results.append((ood_name, auroc, fpr95))
        print(f"  Samples: ID={len(id_scores)}, OOD={len(ood_scores)}")
        print(f"  {ood_name}: AUROC={auroc:.4f}, FPR95={fpr95:.4f}")

    print("\nResults:")
    print("OOD\tAUROC\tFPR95")
    for ood_name, auroc, fpr95 in results:
        print(f"{ood_name}\t{auroc:.4f}\t{fpr95:.4f}")
    if len(results) > 1:
        mean_auroc = float(np.mean([r[1] for r in results]))
        mean_fpr95 = float(np.mean([r[2] for r in results]))
        print(f"Mean\t{mean_auroc:.4f}\t{mean_fpr95:.4f}")
    pipe.remove_hooks()


if __name__ == "__main__":
    main()
