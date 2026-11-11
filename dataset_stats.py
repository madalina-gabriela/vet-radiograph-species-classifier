#!/usr/bin/env python3
"""Print DICOM / patient counts per class (for manuscripts and reports)."""

import argparse
import json
from pathlib import Path

from dataset_utils import compute_dataset_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Count patients and DICOM images per class.")
    parser.add_argument("--data_root", type=str, default="/home/miorga/Doctorat/vet_dataset")
    parser.add_argument("--cats_dir", type=str, default="Pisici")
    parser.add_argument("--dogs_dir", type=str, default="Catei")
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save statistics as JSON.",
    )
    args = parser.parse_args()

    stats = compute_dataset_statistics(
        Path(args.data_root),
        cats_dir=args.cats_dir,
        dogs_dir=args.dogs_dir,
    )

    dog = stats["by_class"]["dog"]
    cat = stats["by_class"]["cat"]

    print("Dataset statistics (patient-level folders)")
    print(f"  Root: {stats['data_root']}")
    print()
    print(f"  Câini ({args.dogs_dir}):")
    print(f"    pacienți:        {dog['patients']}")
    print(f"    imagini DICOM:   {dog['dicom_images']}")
    print()
    print(f"  Pisici ({args.cats_dir}):")
    print(f"    pacienți:        {cat['patients']}")
    print(f"    imagini DICOM:   {cat['dicom_images']}")
    print()
    print(f"  Total pacienți:      {stats['total_patients']}")
    print(f"  Total imagini DICOM: {stats['total_dicom_images']}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON: {out}")


if __name__ == "__main__":
    main()
