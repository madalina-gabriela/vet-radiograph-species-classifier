#!/usr/bin/env python3
"""Training / validation accuracy and loss curves from metrics JSON history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_history(data: Dict[str, Any]) -> Tuple[List[int], List[str], List[float], List[float], List[float], List[float]]:
    epochs: List[int] = []
    stages: List[str] = []
    tr_loss: List[float] = []
    va_loss: List[float] = []
    tr_acc: List[float] = []
    va_acc: List[float] = []
    for i, row in enumerate(data["history"], start=1):
        epochs.append(i)
        stages.append(str(row.get("stage", "")))
        tr_loss.append(float(row["train_loss"]))
        va_loss.append(float(row["val_loss"]))
        tr_acc.append(100.0 * float(row["train_acc"]))
        va_acc.append(100.0 * float(row["val_acc"]))
    return epochs, stages, tr_loss, va_loss, tr_acc, va_acc


def best_val_epoch(va_acc: List[float]) -> int:
    return int(np.argmax(va_acc))


def add_phase_marker(ax, fc_epochs: int = 5) -> None:
    ax.axvline(fc_epochs + 0.5, color="#888", linestyle="--", linewidth=1, alpha=0.75)
    ytop = ax.get_ylim()[1]
    ax.text(fc_epochs / 2, ytop, "FC head", ha="center", va="bottom", fontsize=8, color="#555")
    ax.text(fc_epochs + 0.5 + (len(ax.get_xticks()) or 8) / 6, ytop, "Finetune", ha="center", va="bottom", fontsize=8, color="#555")


def style_axes(ax) -> None:
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_accuracy(epochs, tr_acc, va_acc, out: Path, title: str, mark_best: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    ax.plot(epochs, tr_acc, "o-", color="#1565c0", lw=2, ms=5, label="Training")
    ax.plot(epochs, va_acc, "s-", color="#c62828", lw=2, ms=5, label="Validation")
    if mark_best:
        bi = best_val_epoch(va_acc)
        ax.scatter([epochs[bi]], [va_acc[bi]], s=90, c="#2e7d32", zorder=5, edgecolors="white", lw=1.2)
        ax.annotate(
            f"Selected checkpoint\n(epoch {epochs[bi]}, {va_acc[bi]:.1f}%)",
            xy=(epochs[bi], va_acc[bi]),
            xytext=(epochs[bi] - 2.2, va_acc[bi] - 8),
            fontsize=8,
            color="#2e7d32",
            arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1),
        )
    add_phase_marker(ax)
    ax.set_xlabel("Training epoch (global index)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(epochs)
    ax.set_ylim(max(45, min(tr_acc + va_acc) - 8), 100)
    style_axes(ax)
    ax.legend(loc="lower right")
    ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_loss(epochs, tr_loss, va_loss, out: Path, title: str, mark_best_acc_epoch: int | None = None, va_acc: List[float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    ax.plot(epochs, tr_loss, "o-", color="#1565c0", lw=2, ms=5, label="Training loss")
    ax.plot(epochs, va_loss, "s-", color="#c62828", lw=2, ms=5, label="Validation loss")
    if mark_best_acc_epoch is not None:
        ax.axvline(mark_best_acc_epoch, color="#2e7d32", linestyle=":", lw=1.5, alpha=0.9, label="Best val accuracy epoch")
    add_phase_marker(ax)
    ax.set_xlabel("Training epoch (global index)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xticks(epochs)
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_combined(epochs, tr_loss, va_loss, tr_acc, va_acc, out: Path) -> None:
    bi = best_val_epoch(va_acc)
    best_ep = epochs[bi]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.2), dpi=300, sharex=True)
    ax1.plot(epochs, tr_acc, "o-", color="#1565c0", lw=2, ms=4, label="Train acc.")
    ax1.plot(epochs, va_acc, "s-", color="#c62828", lw=2, ms=4, label="Val acc.")
    ax1.axvline(best_ep, color="#2e7d32", linestyle=":", lw=1.3, alpha=0.85)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(50, 100)
    style_axes(ax1)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.plot(epochs, tr_loss, "o-", color="#1565c0", lw=2, ms=4, label="Train loss")
    ax2.plot(epochs, va_loss, "s-", color="#c62828", lw=2, ms=4, label="Val loss")
    ax2.axvline(best_ep, color="#2e7d32", linestyle=":", lw=1.3, alpha=0.85, label="Best val acc. epoch")
    ax2.set_xlabel("Training epoch (global index; FC head epochs 1–5, then fine-tuning)")
    ax2.set_ylabel("Cross-entropy loss")
    style_axes(ax2)
    ax2.legend(loc="upper right", fontsize=9)

    for ax in (ax1, ax2):
        ax.axvline(5.5, color="#888", linestyle="--", lw=1, alpha=0.7)

    fig.suptitle("ResNet-18 training dynamics (held-out validation; patient-level split, seed 42)", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_interpretation(out: Path, data: Dict[str, Any], epochs, va_acc, va_loss, tr_acc) -> None:
    bi = best_val_epoch(va_acc)
    ep = epochs[bi]
    test_acc = data.get("metrics", {}).get("accuracy")
    text = f"""# Training curve interpretation (reviewer response)

**Source:** `{data.get('seed', '?')}` seed, patient-level split, checkpoint = highest **validation accuracy** (not lowest validation loss).

## Observed dynamics (epochs 1–{epochs[-1]})

- **Training accuracy** rises from {tr_acc[0]:.1f}% to {tr_acc[-1]:.1f}% (fine-tuning phase increases separation from validation).
- **Validation accuracy** is **non-monotonic**: peak **{va_acc[bi]:.1f}%** at global epoch **{ep}**, then decreases to **{va_acc[-1]:.1f}%** by epoch {epochs[-1]} while training accuracy continues to improve → **mild overfitting** after epoch {ep}.
- **Validation loss** increases after epoch {ep} ({va_loss[bi]:.3f} → {va_loss[-1]:.3f}) although training loss falls ({data['history'][0]['train_loss']:.3f} → {data['history'][-1]['train_loss']:.3f}), consistent with growing train–validation gap.

## Conservative reporting

- We report **held-out test** performance (image-level accuracy **{100*test_acc:.1f}%** when available) from the checkpoint at epoch {ep}, **not** the last epoch.
- Validation curves on **n=69** images (28 patients) are noisy; small shifts should not be over-interpreted.
- **Single training run (seed 42)** is shown; multi-seed stability was not re-run for this revision—see limitation paragraph.

## Suggested manuscript wording (English)

> Learning curves showed increasing training accuracy during fine-tuning, while validation accuracy plateaued and then fluctuated (peak validation accuracy {va_acc[bi]:.1f}% at epoch {ep}). Validation loss increased in later fine-tuning epochs despite continued training loss reduction, suggesting mild overfitting; therefore the model checkpoint with the highest validation accuracy was used for test evaluation rather than the final epoch. Performance should be interpreted primarily on the held-out patient-level test set (n=89 images), not on training or validation curves alone.

## Multi-seed runs

Only **one** full history is archived (`metrics_1.json`, seed 42). To add seed stability figures, re-run:

```bash
for s in 42 123 456; do
  python train_catdog.py --seed $s --output_dir Results/seed_$s ...
done
python plot_training_curves.py --multi_glob 'Results/seed_*/metrics.json'
```
"""
    if test_acc is None:
        text = text.replace(" when available", "")
    out.write_text(text, encoding="utf-8")


def plot_multiseed(multiseed_dir: Path, out_dir: Path) -> None:
    runs: List[Dict[str, Any]] = []
    for seed_dir in sorted(multiseed_dir.glob("seed_*")):
        mp = seed_dir / "metrics.json"
        if not mp.exists():
            continue
        data = json.loads(mp.read_text(encoding="utf-8"))
        seed = int(data.get("seed", seed_dir.name.split("_")[-1]))
        mi = data.get("metrics_image_level") or data.get("metrics", {})
        runs.append(
            {
                "seed": seed,
                "best_val_acc": 100 * float(data.get("best_val_acc", 0)),
                "test_acc": 100 * float(mi.get("accuracy", 0)),
                "test_balanced": 100 * float(mi.get("balanced_accuracy", 0)),
                "test_f1_macro": 100 * float(mi.get("f1_macro", 0)),
                "history": data.get("history", []),
            }
        )
    if len(runs) < 2:
        raise SystemExit(f"Need at least 2 seed runs in {multiseed_dir}, found {len(runs)}")

    runs.sort(key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]
    test_accs = [r["test_acc"] for r in runs]
    val_peaks = [r["best_val_acc"] for r in runs]

    summary = {
        "seeds": seeds,
        "test_accuracy_percent": test_accs,
        "best_val_accuracy_percent": val_peaks,
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
        "test_accuracy_min": float(min(test_accs)),
        "test_accuracy_max": float(max(test_accs)),
    }
    (out_dir / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Bar chart — held-out test accuracy
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    x = np.arange(len(seeds))
    bars = ax.bar(x, test_accs, color="#1565c0", edgecolor="#0d47a1", width=0.55)
    if len(test_accs) > 1:
        m, s = float(np.mean(test_accs)), float(np.std(test_accs, ddof=1))
        ax.axhline(m, color="#c62828", ls="--", lw=1.2, label=f"Mean {m:.1f}% ± {s:.1f}%")
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{acc:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Held-out test accuracy (%)")
    ax.set_ylim(0, max(test_accs) + 8)
    ax.set_title("Repeated training runs (same patient-level split)")
    style_axes(ax)
    if len(test_accs) > 1:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_multiseed_test_accuracy.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "figure_multiseed_test_accuracy.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Overlay validation accuracy curves
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    colors = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a"]
    for i, run in enumerate(runs):
        hist = run["history"]
        if not hist:
            continue
        ep = list(range(1, len(hist) + 1))
        va = [100 * float(h["val_acc"]) for h in hist]
        ax.plot(ep, va, "o-", lw=1.8, ms=4, color=colors[i % len(colors)], label=f"Seed {run['seed']} (peak {max(va):.1f}%)")
    ax.axvline(5.5, color="#888", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Training epoch (global index)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation accuracy across random seeds")
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_multiseed_val_accuracy_overlay.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "figure_multiseed_val_accuracy_overlay.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Update interpretation appendix
    lines = [
        "## Multi-seed stability (supplement)",
        "",
        f"Independent training runs with seeds **{', '.join(str(s) for s in seeds)}** "
        f"(identical patient-level train/val/test split; weight initialization and minibatch order differ).",
        "",
        f"- Test accuracy (%): {', '.join(f'{a:.1f}' for a in test_accs)}",
        f"- Mean ± SD: **{summary['test_accuracy_mean']:.2f} ± {summary['test_accuracy_std']:.2f}**",
        f"- Range: {summary['test_accuracy_min']:.1f}–{summary['test_accuracy_max']:.1f}%",
        "",
        "Interpretation: modest run-to-run variation is expected with n=89 test images; "
        "report mean±SD or range alongside the primary seed-42 run.",
        "",
    ]
    interp = out_dir / "training_curves_interpretation.md"
    if interp.exists():
        interp.write_text(interp.read_text(encoding="utf-8") + "\n".join(lines), encoding="utf-8")
    print(f"Multiseed summary: mean test acc {summary['test_accuracy_mean']:.2f}% ± {summary['test_accuracy_std']:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="/home/miorga/Doctorat/test1/metrics_1.json")
    parser.add_argument("--output_dir", default="/home/miorga/Doctorat/test1")
    parser.add_argument(
        "--multiseed_dir",
        default="",
        help="If set, also plot seed stability from test1/multiseed/seed_*/metrics.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    epochs, stages, tr_loss, va_loss, tr_acc, va_acc = parse_history(data)
    bi = best_val_epoch(va_acc)

    plot_accuracy(
        epochs,
        tr_acc,
        va_acc,
        out_dir / "figure_training_validation_accuracy_full.png",
        "Training and validation accuracy (full schedule)",
    )
    plot_loss(
        epochs,
        tr_loss,
        va_loss,
        out_dir / "figure_training_validation_loss.png",
        "Training and validation loss (cross-entropy)",
        mark_best_acc_epoch=epochs[bi],
        va_acc=va_acc,
    )
    plot_combined(epochs, tr_loss, va_loss, tr_acc, va_acc, out_dir / "figure_training_curves_combined.png")
    write_interpretation(out_dir / "training_curves_interpretation.md", data, epochs, va_acc, va_loss, tr_acc)

    # Update classic 10-epoch accuracy figure path
    e10 = [e for e in epochs if e <= 10]
    plot_accuracy(
        e10,
        tr_acc[:10],
        va_acc[:10],
        out_dir / "figure3_training_validation_accuracy.png",
        "Training and validation accuracy (epochs 1–10)",
    )

    print(f"Wrote figures to {out_dir}")
    print(f"Best val acc {va_acc[bi]:.2f}% at epoch {epochs[bi]}")

    if args.multiseed_dir:
        plot_multiseed(Path(args.multiseed_dir), out_dir)


if __name__ == "__main__":
    main()
