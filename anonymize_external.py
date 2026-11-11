#!/usr/bin/env python3
"""
Copy and anonymize external-clinic JPEG radiographs for pilot external validation.

Anonymization: strip EXIF, rename to ext_XXX.jpg, crop margins to remove burned-in
clinic/patient overlays (best-effort; authors should visually QC every image).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def crop_anonymize(img: Image.Image, margins: Tuple[float, float, float, float]) -> Image.Image:
    """Margins as fractions (top, bottom, left, right) to remove overlay text."""
    w, h = img.size
    top, bottom, left, right = margins
    x0 = int(w * left)
    x1 = int(w * (1 - right))
    y0 = int(h * top)
    y1 = int(h * (1 - bottom))
    cropped = img.crop((x0, y0, x1, y1))
    return cropped.convert("L")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/home/miorga/Downloads/Fwd_ rx/test",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/miorga/Doctorat/vet_external_validation",
    )
    parser.add_argument("--top", type=float, default=0.17)
    parser.add_argument("--bottom", type=float, default=0.24)
    parser.add_argument("--left", type=float, default=0.02)
    parser.add_argument("--right", type=float, default=0.14)
    args = parser.parse_args()

    source = Path(args.source_dir)
    out_root = Path(args.output_root)
    raw_dir = out_root / "raw_copy"
    anon_dir = out_root / "anonymized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    anon_dir.mkdir(parents=True, exist_ok=True)

    margins = (args.top, args.bottom, args.left, args.right)
    files = sorted(source.glob("*.jpeg")) + sorted(source.glob("*.jpg")) + sorted(source.glob("*.JPEG"))

    manifest: List[Dict[str, str]] = []
    for i, src in enumerate(files, start=1):
        ext_id = f"ext_{i:03d}"
        raw_copy = raw_dir / f"{ext_id}_original{src.suffix.lower()}"
        shutil.copy2(src, raw_copy)

        img = Image.open(src)
        img.load()
        anon = crop_anonymize(img, margins)
        out_path = anon_dir / f"{ext_id}.jpg"
        anon.save(out_path, format="JPEG", quality=95)

        manifest.append(
            {
                "ext_id": ext_id,
                "anonymized_path": str(out_path),
                "raw_copy_path": str(raw_copy),
                "source_filename": src.name,
            }
        )

    (out_root / "anonymization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = out_root / "README.txt"
    readme.write_text(
        "External validation pilot set (n=10) from a second veterinary clinic.\n"
        "Images are JPEG exports/thumbnails, not DICOM.\n"
        "Anonymized copies: margin crop + rename; EXIF stripped on save.\n"
        "Ground-truth labels: labels.csv (from clinic species at transfer — verify before publication).\n"
        "Do not share raw_copy/ in public repositories.\n",
        encoding="utf-8",
    )
    print(f"Processed {len(manifest)} images -> {anon_dir}")


if __name__ == "__main__":
    main()
