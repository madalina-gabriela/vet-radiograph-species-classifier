#!/usr/bin/env python3
"""Identify test patients misclassified across M0–M7, M9, M10 and summarize DICOM metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataset_characteristics import read_image_meta
from dataset_utils import build_patient_records, load_split_from_json, patient_key_for_image

DEFAULT_CONFIGS = [f"M{i}" for i in range(8)] + ["M9", "M10"]


def load_patient_preds(comparison_dir: Path, code: str) -> dict:
    path = comparison_dir / code / "best" / "predictions_patient_level.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["patient_key"]: r for r in rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--comparison_dir", type=str, default="/home/miorga/Doctorat/test_comparison")
    p.add_argument("--data_root", type=str, default="/home/miorga/Doctorat/vet_dataset")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    comp = Path(args.comparison_dir)
    data_root = Path(args.data_root)
    configs = DEFAULT_CONFIGS
    base = load_patient_preds(comp, "M0")
    patients = sorted(base.keys())

    wrong_count: Counter = Counter()
    for pk in patients:
        for code in configs:
            if not load_patient_preds(comp, code)[pk]["correct"]:
                wrong_count[pk] += 1

    always_wrong = [pk for pk in patients if wrong_count[pk] == len(configs)]
    records = {r.patient_key: r for r in build_patient_records(data_root)}

    out_lines = [
        f"Test patients: {len(patients)}",
        f"Wrong on all {len(configs)} configs: {len(always_wrong)}",
        "",
    ]
    for pk in always_wrong:
        rec = records[pk]
        proj = Counter()
        reg = Counter()
        for img in rec.image_paths:
            m = read_image_meta(img, pk, rec.label)
            if m:
                proj[m.projection] += 1
                reg[m.region_category] += 1
        row = base[pk]
        out_lines.append(
            f"{pk} | true={row['true_label_name']} | n_img={row['n_images']} | "
            f"regions={dict(reg)} | projections={dict(proj)} | M0 prob_cat={row['prob_cat']:.3f}"
        )

    text = "\n".join(out_lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
