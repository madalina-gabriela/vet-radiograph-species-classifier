#!/usr/bin/env python3
"""Pick best LR run by validation macro-F1 within one experiment folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def collect_runs(exp_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("lr_"):
            continue
        mp = sub / "metrics.json"
        if not mp.exists():
            continue
        data = json.loads(mp.read_text(encoding="utf-8"))
        runs.append(
            {
                "run_dir": str(sub),
                "lr": float(data.get("protocol", {}).get("lr", data.get("lr", 0))),
                "best_val_macro_f1": float(data.get("best_val_macro_f1", -1)),
                "best_val_acc": float(data.get("best_val_acc", 0)),
                "test_accuracy": float((data.get("metrics_image_level") or data.get("metrics", {})).get("accuracy", 0)),
                "training_time_seconds": float(data.get("training_time_seconds", 0)),
                "selected_epoch": int(data.get("selected_epoch", 0)),
            }
        )
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True, help="Experiment folder, e.g. test_comparison/M0")
    p.add_argument("--experiment", type=str, default="", help="Optional label (M0)")
    args = p.parse_args()

    exp_dir = Path(args.dir)
    runs = collect_runs(exp_dir)
    if not runs:
        raise SystemExit(f"No lr_* runs with metrics.json under {exp_dir}")

    best = max(runs, key=lambda r: r["best_val_macro_f1"])
    out = {
        "experiment": args.experiment or exp_dir.name,
        "selected_lr": best["lr"],
        "selected_run_dir": best["run_dir"],
        "best_val_macro_f1": best["best_val_macro_f1"],
        "test_accuracy": best["test_accuracy"],
        "selected_epoch": best["selected_epoch"],
        "training_time_seconds": best["training_time_seconds"],
        "all_runs": runs,
    }
    (exp_dir / "best_lr_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    best_link = exp_dir / "best"
    if best_link.exists():
        if best_link.is_symlink():
            best_link.unlink()
        else:
            shutil.rmtree(best_link)
    best_link.symlink_to(Path(best["run_dir"]).resolve(), target_is_directory=True)

    print(f"Best LR: {best['lr']} (val macro-F1={best['best_val_macro_f1']:.4f})")
    print(f"Linked: {best_link} -> {best['run_dir']}")
    print(f"Wrote {exp_dir / 'best_lr_selection.json'}")


if __name__ == "__main__":
    main()
