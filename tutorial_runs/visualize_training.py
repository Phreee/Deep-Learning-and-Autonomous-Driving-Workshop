import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _read_train_log_csv(path: str) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    epochs: List[int] = []
    train_loss: List[float] = []
    train_acc: List[float] = []
    val_loss: List[float] = []
    val_acc: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
    return epochs, train_loss, train_acc, val_loss, val_acc


def _read_metrics(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training logs (CSV + metrics.json)")
    parser.add_argument(
        "--exp_dirs",
        nargs="+",
        default=None,
        help="Experiment directories (each must include train_log.csv + metrics.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison figure",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root = os.path.join(base_dir, "tutorial_runs", "output")
    default_plot = os.path.join(out_root, "training_comparison.png")
    plot_path = args.output or default_plot

    if args.exp_dirs is None:
        args.exp_dirs = [
            os.path.join(out_root, "simple_cnn"),
            os.path.join(out_root, "resnet18"),
        ]

    series = []
    for exp in args.exp_dirs:
        log_path = os.path.join(exp, "train_log.csv")
        metrics_path = os.path.join(exp, "metrics.json")
        if not os.path.exists(log_path) or not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"Missing train_log.csv or metrics.json in {exp}. "
                "Run training first with cifar_cnn_resnet.py"
            )
        metrics = _read_metrics(metrics_path)
        epochs, tr_l, tr_a, va_l, va_a = _read_train_log_csv(log_path)
        series.append(
            {
                "name": metrics.get("model", os.path.basename(exp)),
                "epochs": epochs,
                "train_loss": tr_l,
                "train_acc": tr_a,
                "val_loss": va_l,
                "val_acc": va_a,
                "metrics": metrics,
            }
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for s in series:
        ax1.plot(s["epochs"], s["train_loss"], label=f"{s['name']} train", linewidth=2)
        ax1.plot(s["epochs"], s["val_loss"], label=f"{s['name']} val", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss 对比", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for s in series:
        ax2.plot(s["epochs"], s["train_acc"], label=f"{s['name']} train", linewidth=2)
        ax2.plot(s["epochs"], s["val_acc"], label=f"{s['name']} val", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy 对比", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图表到: {plot_path}")

    print("\n" + "=" * 60)
    print("训练结果总结")
    print("=" * 60)
    for s in series:
        m = s["metrics"]
        gap = float(m.get("final_train_acc", 0.0)) - float(m.get("final_val_acc", 0.0))
        print(f"\n{s['name']}:")
        print(f"  最终训练准确率: {m.get('final_train_acc'):.4f}")
        print(f"  最终验证准确率: {m.get('final_val_acc'):.4f}")
        print(f"  过拟合程度(train-val): {gap:.4f}")
        print(f"  训练时间: {m.get('elapsed_sec')} 秒")
        print(f"  参数量: {m.get('num_params')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
