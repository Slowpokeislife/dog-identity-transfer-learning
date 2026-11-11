"""Two-stage training and held-out validation."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import build_model, configure_trainable, preprocess


def get_loaders(data_dir: Path, batch_size: int, workers: int):
    normalize = transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225))
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15),
        transforms.ToTensor(), normalize,
    ])
    valid_transform = preprocess()
    datasets_by_split = {}
    for split, tfm in (("train", train_transform), ("valid", valid_transform)):
        ds = datasets.ImageFolder(data_dir / split, transform=tfm)
        if ds.classes != ["bo", "not_bo"]:
            raise ValueError(f"Expected bo/ and not_bo/ class folders in {split}; found {ds.classes}")
        if not all(ds.targets.count(i) > 0 for i in (0, 1)):
            raise ValueError(f"Both classes must have images in {split}")
        datasets_by_split[split] = ds
    return ({split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                               num_workers=workers, pin_memory=torch.cuda.is_available())
             for split, ds in datasets_by_split.items()},
            {split: len(ds) for split, ds in datasets_by_split.items()})


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    tp = tn = fp = fn = 0
    for images, labels in loader:
        images = images.to(device)
        # ImageFolder bo=0, not_bo=1, matching positive sigmoid output.
        target = labels.to(device).float()
        with torch.set_grad_enabled(training):
            logits = model(images).flatten()
            loss = criterion(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        pred = (torch.sigmoid(logits) >= .5).long()
        gold = target.long()
        n = gold.numel()
        total_loss += loss.item() * n
        total += n
        correct += (pred == gold).sum().item()
        tp += ((pred == 0) & (gold == 0)).sum().item()  # Bo as positive for metrics
        tn += ((pred == 1) & (gold == 1)).sum().item()
        fp += ((pred == 0) & (gold == 1)).sum().item()
        fn += ((pred == 1) & (gold == 0)).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"loss": round(total_loss / total, 6), "accuracy": round(correct / total, 6),
            "precision_bo": round(precision, 6), "recall_bo": round(recall, 6),
            "f1_bo": round(f1, 6), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "samples": total}


def main():
    parser = argparse.ArgumentParser(description="Train a VGG16 image identity classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/presidential_doggy_door"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-fine-tune", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-fine-tune", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs_head < 1 or args.epochs_fine_tune < 0:
        parser.error("epochs-head must be >=1 and epochs-fine-tune must be >=0")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, sizes = get_loaders(args.data_dir, args.batch_size, args.num_workers)
    print(f"Device: {device} | Dataset sizes: {sizes}")
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    for phase, epochs, lr in (("head", args.epochs_head, args.lr_head),
                              ("fine_tune", args.epochs_fine_tune, args.lr_fine_tune)):
        if epochs == 0: continue
        configure_trainable(model, phase)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
            valid_metrics = run_epoch(model, loaders["valid"], criterion, device)
            history.append({"phase": phase, "epoch": epoch, "train_loss": train_metrics["loss"],
                            "train_accuracy": train_metrics["accuracy"],
                            **{f"valid_{k}": v for k, v in valid_metrics.items()}})
            print(f"{phase} {epoch}/{epochs}: train acc {train_metrics['accuracy']:.3f} | "
                  f"val acc {valid_metrics['accuracy']:.3f} | val loss {valid_metrics['loss']:.3f}")
            if valid_metrics["loss"] < best_val_loss:
                best_val_loss = valid_metrics["loss"]
                torch.save({"model_state": model.state_dict(), "classes": ["bo", "not_bo"],
                            "architecture": "vgg16", "threshold": .5, "phase": phase},
                           args.output_dir / "best_model.pt")
    with (args.output_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    with (args.output_dir / "run_config.json").open("w") as fh:
        json.dump({**vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir),
                   "device": str(device), "dataset_sizes": sizes}, fh, indent=2)
    print("Best checkpoint:", args.output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
