import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    import torchvision
    from torchvision import transforms
except ImportError as exc:
    raise SystemExit("Missing torchvision. Install with: python -m pip install torchvision") from exc


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class EpochRow:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


@dataclass
class RunMetrics:
    model: str
    num_params: int
    device: str
    num_epochs: int
    train_subset: int
    test_subset: int
    batch_size: int
    lr: float
    weight_decay: float
    data_augment: bool
    elapsed_sec: float
    final_train_acc: float
    final_val_acc: float
    final_train_loss: float
    final_val_loss: float


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / max(total, 1)
    return avg_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / max(total, 1)
    return avg_loss, acc


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cifar10_transforms(data_augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfms: List[transforms.Compose] = []
    if data_augment:
        train_tfms.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_tfms.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(train_tfms), test_tfms


def _save_train_log_csv(rows: List[EpochRow], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _plot_training(rows: List[EpochRow], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r.epoch for r in rows]
    train_loss = [r.train_loss for r in rows]
    val_loss = [r.val_loss for r in rows]
    train_acc = [r.train_acc for r in rows]
    val_acc = [r.val_acc for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, train_loss, label="train_loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="val_loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_title("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_acc, label="train_acc", linewidth=2)
    ax2.plot(epochs, val_acc, label="val_acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_title("Accuracy")
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    *,
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    data_augment: bool,
    train_subset: int,
    test_subset: int,
    batch_size: int,
) -> Dict:
    _ensure_dir(out_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[EpochRow] = []
    start = time.time()

    print(f"\n开始训练 {name}...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, test_loader, criterion, device)
        history.append(
            EpochRow(
                epoch=epoch + 1,
                train_loss=round(train_loss, 6),
                train_acc=round(train_acc, 6),
                val_loss=round(val_loss, 6),
                val_acc=round(val_acc, 6),
            )
        )
        print(
            f"Epoch {epoch+1}/{num_epochs} - train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
        )

    elapsed = time.time() - start
    num_params = _count_params(model)

    # Save artifacts (match tutorial 2.3)
    train_log_path = os.path.join(out_dir, "train_log.csv")
    model_path = os.path.join(out_dir, "model.pth")
    plot_path = os.path.join(out_dir, "training_plot.png")
    metrics_path = os.path.join(out_dir, "metrics.json")

    _save_train_log_csv(history, train_log_path)
    torch.save(model.state_dict(), model_path)
    _plot_training(history, plot_path)

    last = history[-1]
    metrics = RunMetrics(
        model=name,
        num_params=num_params,
        device=str(device),
        num_epochs=num_epochs,
        train_subset=train_subset,
        test_subset=test_subset,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        data_augment=data_augment,
        elapsed_sec=round(elapsed, 2),
        final_train_acc=last.train_acc,
        final_val_acc=last.val_acc,
        final_train_loss=last.train_loss,
        final_val_loss=last.val_loss,
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)

    return {
        "out_dir": out_dir,
        "train_log": train_log_path,
        "model_path": model_path,
        "training_plot": plot_path,
        "metrics": metrics_path,
        "metrics_obj": asdict(metrics),
        "history": [asdict(r) for r in history],
    }


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10: Simple CNN vs ResNet18")
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=["simple_cnn", "resnet18", "both"],
        help="Which model to train",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--subset", type=int, default=5000, help="Train subset size")
    parser.add_argument("--test_subset", type=int, default=1000, help="Test subset size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--data_augment",
        type=str,
        default="False",
        help="True/False. Enable basic data augmentation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for this experiment (e.g. tutorial_runs/output/simple_cnn)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_augment = str(args.data_augment).lower() in {"1", "true", "yes", "y"}

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out_root = os.path.join(base_dir, "tutorial_runs", "output")
    out_dir = args.output or os.path.join(default_out_root, args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(args.seed)

    train_tfm, test_tfm = _cifar10_transforms(data_augment=data_augment)
    data_root = os.path.join(base_dir, "tutorial_runs", "data")
    train_ds = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_tfm,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_tfm,
    )

    train_subset = Subset(train_ds, list(range(min(args.subset, len(train_ds)))))
    test_subset = Subset(test_ds, list(range(min(args.test_subset, len(test_ds)))))
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def _make_model(model_name: str) -> nn.Module:
        if model_name == "simple_cnn":
            return SimpleCNN(num_classes=10)
        if model_name == "resnet18":
            return torchvision.models.resnet18(weights=None, num_classes=10)
        raise ValueError(f"Unknown model: {model_name}")

    runs: List[Dict] = []
    models_to_run: List[str]
    if args.model == "both":
        models_to_run = ["simple_cnn", "resnet18"]
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        this_out = out_dir
        if args.model == "both":
            this_out = os.path.join(out_dir, model_name)

        model = _make_model(model_name).to(device)
        run = run_experiment(
            name=model_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            out_dir=this_out,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            data_augment=data_augment,
            train_subset=len(train_subset),
            test_subset=len(test_subset),
            batch_size=args.batch_size,
        )
        runs.append(run)
        print("saved", run["train_log"])
        print("saved", run["model_path"])
        print("saved", run["training_plot"])
        print("saved", run["metrics"])

    # Optional: write a combined summary if multiple models are run
    if len(runs) > 1:
        _ensure_dir(default_out_root)
        summary_path = os.path.join(default_out_root, "cnn_resnet_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "model": r["metrics_obj"]["model"],
                        "metrics": r["metrics_obj"],
                        "history": r["history"],
                        "out_dir": r["out_dir"],
                    }
                    for r in runs
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("saved", summary_path)


if __name__ == "__main__":
    main()
