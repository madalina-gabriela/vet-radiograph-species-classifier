#!/usr/bin/env python3
"""Bar charts comparing completed M0–M8 experiments (uses each M*/best/ run)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from experiment_registry import EXPERIMENTS


def load_best(exp_root: Path) -> Dict[str, Any] | None:
    sel = exp_root / "best_lr_selection.json"
    metrics_path = exp_root / "best" / "metrics.json"
    if not metrics_path.exists():
        mp = exp_root / "best" / "metrics.json"
        if not mp.exists():
            for p in exp_root.glob("lr_*/metrics.json"):
                metrics_path = p
                break
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    mi = data.get("metrics_image_level") or data.get("metrics", {})
    exp_code = data.get("experiment", {}).get("code") or exp_root.name
    meta = EXPERIMENTS.get(exp_code, {})
    return {
        "code": exp_code,
        "title": meta.get("title", exp_code),
        "test_acc": 100 * float(mi.get("accuracy", 0)),
        "test_bal": 100 * float(mi.get("balanced_accuracy", 0)),
        "test_f1": 100 * float(mi.get("f1_macro", 0)),
        "val_macro_f1": 100 * float(data.get("best_val_macro_f1", 0)),
        "selected_epoch": int(data.get("selected_epoch", 0)),
        "train_time_s": float(data.get("training_time_seconds", 0)),
        "lr": data.get("protocol", {}).get("lr"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--comparison_dir", type=str, default="/home/miorga/Doctorat/test_comparison")
    p.add_argument("--output", type=str, default="")
    p.add_argument(
        "--codes",
        type=str,
        default="",
        help="Comma-separated experiment codes (default: all M* with best/metrics.json, registry order)",
    )
    args = p.parse_args()
    root = Path(args.comparison_dir)
    if args.codes.strip():
        wanted = [c.strip().upper() for c in args.codes.split(",") if c.strip()]
    else:
        wanted = sorted(EXPERIMENTS.keys())

    rows: List[Dict[str, Any]] = []
    for code in wanted:
        if code not in EXPERIMENTS:
            raise SystemExit(f"Unknown experiment code {code!r}")
        exp_dir = root / code
        if not exp_dir.is_dir():
            continue
        row = load_best(exp_dir)
        if row:
            rows.append(row)
        else:
            print(f"Warning: skipping {code} (no best/metrics.json)")

    if not rows:
        raise SystemExit(f"No completed experiments under {root} (need M*/best/metrics.json)")

    out_dir = Path(args.output) if args.output else root
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [r["code"] for r in rows]
    x = np.arange(len(rows))
    accs = [r["test_acc"] for r in rows]
    f1s = [r["test_f1"] for r in rows]

    fig_h = 5.2
    fig_w = max(9.0, len(rows) * 0.95)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    w = 0.36
    bars_acc = ax.bar(x - w / 2, accs, w, label="Test accuracy", color="#1565c0")
    bars_f1 = ax.bar(x + w / 2, f1s, w, label="Test macro-F1", color="#c62828")
    ax.bar_label(bars_acc, fmt="%.1f", padding=2, fontsize=8, fontweight="bold")
    ax.bar_label(bars_f1, fmt="%.1f", padding=2, fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("Quality metrics (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Experiment", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Held-out test (best LR per experiment, image level)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_comparison_test_metrics.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "figure_comparison_test_metrics.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {"experiments": rows}
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'figure_comparison_test_metrics.png'} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
