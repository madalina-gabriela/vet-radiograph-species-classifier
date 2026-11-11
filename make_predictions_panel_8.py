#!/usr/bin/env python3
"""8-image prediction panel (4 correct + 4 errors) for manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from dicom_preprocess import dicom_to_uint8


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["correct"] = row["correct"].strip().lower() == "true"
            row["true_label_id"] = int(row["true_label_id"])
            row["pred_label_id"] = int(row["pred_label_id"])
            row["prob_dog"] = float(row["prob_dog"])
            row["prob_cat"] = float(row["prob_cat"])
            rows.append(row)
    return rows


def _float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def _bool(row: Dict[str, Any]) -> bool:
    return bool(row["correct"])


def pick_two(
    pool: List[Dict[str, Any]],
    sort_key,
    reverse: bool = True,
) -> List[Dict[str, Any]]:
    """Pick up to 2 rows, prefer distinct patient_key."""
    ordered = sorted(pool, key=sort_key, reverse=reverse)
    chosen: List[Dict[str, Any]] = []
    seen = set()
    for row in ordered:
        pk = row["patient_key"]
        if pk in seen:
            continue
        chosen.append(row)
        seen.add(pk)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        for row in ordered:
            if row not in chosen:
                chosen.append(row)
            if len(chosen) == 2:
                break
    return chosen[:2]


def select_examples(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    correct_dog = [r for r in rows if _bool(r) and r["true_label_name"] == "dog"]
    correct_cat = [r for r in rows if _bool(r) and r["true_label_name"] == "cat"]
    wrong_dog = [r for r in rows if not _bool(r) and r["true_label_name"] == "dog"]
    wrong_cat = [r for r in rows if not _bool(r) and r["true_label_name"] == "cat"]

    conf = lambda r: max(_float(r, "prob_dog"), _float(r, "prob_cat"))

    return {
        "correct_dog": pick_two(correct_dog, conf),
        "correct_cat": pick_two(correct_cat, conf),
        "wrong_dog_as_cat": pick_two(wrong_dog, lambda r: _float(r, "prob_cat")),
        "wrong_cat_as_dog": pick_two(wrong_cat, lambda r: _float(r, "prob_dog")),
    }


def draw_panel(
    groups: Dict[str, List[Dict[str, Any]]],
    out_path: Path,
    dpi: int = 200,
) -> None:
    order = [
        ("Correct — canine", groups["correct_dog"], "#2e7d32"),
        ("Correct — feline", groups["correct_cat"], "#2e7d32"),
        ("Error — canine predicted as feline", groups["wrong_dog_as_cat"], "#c62828"),
        ("Error — feline predicted as canine", groups["wrong_cat_as_dog"], "#c62828"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    fig.suptitle("Representative test-set predictions (ResNet-18)", fontsize=14, fontweight="bold", y=0.98)

    col_titles = ["Canine (true)", "Canine (true)", "Feline (true)", "Feline (true)"]
    row0 = order[0][1] + order[1][1]  # 2 dog + 2 cat correct - reorder to dog, dog, cat, cat
    row0 = groups["correct_dog"][:2] + groups["correct_cat"][:2]
    row1 = groups["wrong_dog_as_cat"][:2] + groups["wrong_cat_as_dog"][:2]

    for row_idx, (row_rows, row_label, color) in enumerate(
        [
            (row0, "Correct predictions", "#2e7d32"),
            (row1, "Incorrect predictions", "#c62828"),
        ]
    ):
        fig.text(0.02, 0.72 - row_idx * 0.46, row_label, fontsize=11, fontweight="bold", color=color, rotation=90, va="center")

    for row_idx, row_rows in enumerate([row0, row1]):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(row_rows):
                continue
            row = row_rows[col_idx]
            img = dicom_to_uint8(Path(row["image_path"]))
            ax.imshow(img, cmap="gray")
            ok = _bool(row)
            edge = "#2e7d32" if ok else "#c62828"
            rect = Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                edgecolor=edge,
                linewidth=3,
            )
            ax.add_patch(rect)
            status = "Correct" if ok else "Wrong"
            ax.set_title(
                f"{status}\n"
                f"True: {row['true_label_name']}  →  Pred: {row['pred_label_name']}\n"
                f"P(dog)={_float(row,'prob_dog'):.2f}  P(cat)={_float(row,'prob_cat'):.2f}",
                fontsize=9,
            )

    if row_idx == 0:
        pass
    fig.tight_layout(rect=[0.04, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/miorga/Doctorat/test1/predictions_detailed.csv")
    parser.add_argument("--output", default="/home/miorga/Doctorat/test1/predictions_panel_8_manuscript.png")
    parser.add_argument("--manifest", default="/home/miorga/Doctorat/test1/predictions_panel_8_manifest.json")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    groups = select_examples(rows)
    for k, v in groups.items():
        if len(v) < 2:
            raise SystemExit(f"Not enough examples for {k}: found {len(v)}")

    out = Path(args.output)
    draw_panel(groups, out)

    flat = []
    for name, items in groups.items():
        for r in items:
            flat.append(
                {
                    "group": name,
                    "image_path": r["image_path"],
                    "patient_key": r["patient_key"],
                    "true": r["true_label_name"],
                    "pred": r["pred_label_name"],
                    "prob_dog": _float(r, "prob_dog"),
                    "prob_cat": _float(r, "prob_cat"),
                }
            )
    Path(args.manifest).write_text(json.dumps(flat, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
