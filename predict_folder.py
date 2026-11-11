#!/usr/bin/env python3
"""Batch inference on a folder of DICOM (.dcm) and optional JPEG files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from dicom_preprocess import dicom_to_uint8
from model_utils import build_resnet18_catdog, infer_fc_dropout, load_checkpoint
from train_catdog import make_transforms


def load_image_uint8(path: Path) -> Image.Image:
    if path.suffix.lower() == ".dcm":
        arr = dicom_to_uint8(path)
        return Image.fromarray(arr, mode="L")
    return Image.open(path).convert("L")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default="/home/miorga/Doctorat/vet1_catdog/results_vet_dataset/best_model.pt",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no_tta", action="store_false", dest="tta")
    parser.add_argument("--prefer_dcm", action="store_true", default=True)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    dropout = float(ckpt_meta.get("fc_dropout") or infer_fc_dropout(ckpt_meta["model_state_dict"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_catdog(fc_dropout=dropout)
    load_checkpoint(model, Path(args.checkpoint))
    model.to(device).eval()
    _, eval_t = make_transforms(224)

    paths: List[Path] = []
    if args.prefer_dcm:
        paths = sorted(in_dir.glob("*.dcm"))
        if not paths:
            paths = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.jpeg"))
    else:
        paths = sorted(in_dir.glob("*"))

    rows = []
    failed = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            pil = load_image_uint8(p)
            x = eval_t(pil).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            if args.tta:
                xf = torch.flip(x, dims=[-1])
                probs = (probs + torch.softmax(model(xf), dim=1)[0].cpu().numpy()) / 2
            pred = int(np.argmax(probs))
            rows.append(
                {
                    "filename": p.name,
                    "path": str(p),
                    "pred_label_id": pred,
                    "pred_label_name": "cat" if pred == 1 else "dog",
                    "prob_cat": float(probs[1]),
                    "prob_dog": float(probs[0]),
                }
            )
        except Exception as e:
            failed.append({"path": str(p), "error": str(e)})

    n_dog = sum(1 for r in rows if r["pred_label_id"] == 0)
    n_cat = sum(1 for r in rows if r["pred_label_id"] == 1)
    summary = {
        "input_dir": str(in_dir),
        "n_processed": len(rows),
        "n_failed": len(failed),
        "pred_dog": n_dog,
        "pred_cat": n_cat,
        "checkpoint": args.checkpoint,
        "tta": args.tta,
    }

    (out_dir / "predictions_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if failed:
        (out_dir / "failed.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nMost cat-like (top 10 P(cat)):")
    for r in sorted(rows, key=lambda x: x["prob_cat"], reverse=True)[:10]:
        print(f"  {r['prob_cat']:.3f} {r['pred_label_name']:3s}  {r['filename']}")
    print("\nMost dog-like (bottom 5 P(cat)):")
    for r in sorted(rows, key=lambda x: x["prob_cat"])[:5]:
        print(f"  {r['prob_cat']:.3f} {r['pred_label_name']:3s}  {r['filename']}")


if __name__ == "__main__":
    main()
