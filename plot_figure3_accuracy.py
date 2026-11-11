#!/usr/bin/env python3
"""Figure 3: training and validation accuracy vs epoch (from metrics JSON history)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(metrics_path: Path, max_epochs: int | None) -> tuple[list[int], list[float], list[float], float]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = data["history"]
    epochs: list[int] = []
    train_acc: list[float] = []
    val_acc: list[float] = []

    global_ep = 0
    for row in history:
        global_ep += 1
        if max_epochs is not None and global_ep > max_epochs:
            break
        epochs.append(global_ep)
        train_acc.append(100.0 * row["train_acc"])
        val_acc.append(100.0 * row["val_acc"])

    best_val = max(val_acc) if val_acc else 0.0
    return epochs, train_acc, val_acc, best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="/home/miorga/Doctorat/test1/metrics_1.json",
    )
    parser.add_argument(
        "--output",
        default="/home/miorga/Doctorat/test1/figure3_training_validation_accuracy.png",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to plot (default 10, manuscript caption).",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs, train_acc, val_acc, best_val = load_history(metrics_path, args.epochs)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(epochs, train_acc, color="#1565c0", linewidth=2, marker="o", markersize=5, label="Training accuracy")
    ax.plot(epochs, val_acc, color="#c62828", linewidth=2, marker="s", markersize=5, label="Validation accuracy")

    best_i = val_acc.index(max(val_acc))
    ax.scatter([epochs[best_i]], [val_acc[best_i]], s=80, color="#2e7d32", zorder=5, edgecolors="white", linewidths=1.2)
    ax.annotate(
        f"Best val {val_acc[best_i]:.1f}%",
        xy=(epochs[best_i], val_acc[best_i]),
        xytext=(epochs[best_i] + 0.35, val_acc[best_i] - 4),
        fontsize=9,
        color="#2e7d32",
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1),
    )

    if args.epochs >= 6:
        ax.axvline(5.5, color="#888", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(3, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] else 99, "FC head", ha="center", va="top", fontsize=8, color="#666")
        ax.text(8, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] else 99, "Finetune", ha="center", va="top", fontsize=8, color="#666")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(epochs)
    ax.set_ylim(50, 100)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"Wrote {pdf_path}")
    print(f"Plotted {len(epochs)} epochs; peak validation accuracy {best_val:.2f}%")


if __name__ == "__main__":
    main()
