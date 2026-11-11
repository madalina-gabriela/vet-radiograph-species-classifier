#!/usr/bin/env python3
"""
External inference with domain-shift mitigations:
  - lighter anonymization crop (optional)
  - percentile intensity scaling (matches DICOM cache)
  - TTA (horizontal flip)
  - decision threshold tuned on internal VALIDATION split (not external labels)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from anonymize_external import crop_anonymize
from external_validate import main as external_validate_main  # noqa: F401
from model_utils import build_resnet18_catdog, infer_fc_dropout, load_checkpoint
from train_catdog import make_transforms


def uint8_from_gray_array(arr: np.ndarray, mode: str) -> np.ndarray:
    arr = arr.astype(np.float32)
    if mode == "percentile":
        lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
        if hi <= lo + 1e-6:
            lo, hi = float(arr.min()), float(arr.max())
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-6) * 255.0
    elif mode == "invert":
        arr = 255.0 - arr
    return arr.astype(np.uint8)


def load_pil_for_external(path: Path, source_jpeg: Path | None, light_margins: Tuple[float, float, float, float], prep: str) -> Image.Image:
    if source_jpeg is not None and source_jpeg.exists():
        rgb = Image.open(source_jpeg).convert("RGB")
        gray = crop_anonymize(rgb, light_margins)
        arr = np.asarray(gray, dtype=np.float32)
    else:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    if prep == "autocontrast":
        return ImageOps.autocontrast(Image.fromarray(arr.astype(np.uint8), mode="L"))
    arr = uint8_from_gray_array(arr, prep if prep != "autocontrast" else "raw")
    return Image.fromarray(arr, mode="L")


@torch.inference_mode()
def predict_prob_cat(model: torch.nn.Module, pil: Image.Image, eval_t, device: torch.device, tta: bool) -> float:
    x = eval_t(pil).unsqueeze(0).to(device)
    p = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    if tta:
        p2 = torch.softmax(model(torch.flip(x, dims=[-1])), dim=1)[0].cpu().numpy()
        p = (p + p2) * 0.5
    return float(p[1])


def tune_threshold_on_internal_val(val_csv: Path) -> float:
    y_true: List[int] = []
    prob_cat: List[float] = []
    for row in csv.DictReader(val_csv.open(encoding="utf-8")):
        y_true.append(int(row["true_label_id"]))
        prob_cat.append(float(row["prob_cat"]))
    yt = np.asarray(y_true)
    pc = np.asarray(prob_cat)
    best_t = 0.5
    best_ba = -1.0
    for t in np.unique(np.round(pc, 4)):
        pred = (pc >= t).astype(int)
        dog_ok = float(np.mean(pred[yt == 0] == 0)) if np.any(yt == 0) else 0.0
        cat_ok = float(np.mean(pred[yt == 1] == 1)) if np.any(yt == 1) else 0.0
        ba = 0.5 * (dog_ok + cat_ok)
        if ba > best_ba:
            best_ba = ba
            best_t = float(t)
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="/home/miorga/Doctorat/vet_external_validation/anonymized")
    parser.add_argument("--source_dir", default="/home/miorga/Downloads/Fwd_ rx/test")
    parser.add_argument("--labels_csv", default="/home/miorga/Doctorat/vet_external_validation/labels.csv")
    parser.add_argument("--manifest_json", default="/home/miorga/Doctorat/vet_external_validation/anonymization_manifest.json")
    parser.add_argument("--checkpoint", default="/home/miorga/Doctorat/vet1_catdog/results_vet_dataset/best_model.pt")
    parser.add_argument("--internal_val_predictions", default="/home/miorga/Doctorat/vet1_catdog/results_vet_dataset/predictions_detailed.csv")
    parser.add_argument("--output_dir", default="/home/miorga/Doctorat/vet_external_validation/results_adapted")
    parser.add_argument("--prep", choices=("raw", "percentile", "invert", "autocontrast"), default="percentile")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no_tta", action="store_false", dest="tta")
    parser.add_argument("--use_light_crop_from_source", action="store_true", default=True)
    parser.add_argument("--crop_top", type=float, default=0.10)
    parser.add_argument("--crop_bottom", type=float, default=0.16)
    parser.add_argument("--crop_left", type=float, default=0.02)
    parser.add_argument("--crop_right", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=-1.0, help="If <0, tune on internal val predictions CSV.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    source_by_ext = {m["ext_id"]: Path("/home/miorga/Downloads/Fwd_ rx/test") / m["source_filename"] for m in manifest}

    labels = {r["ext_id"]: int(r["true_label_id"]) for r in csv.DictReader(Path(args.labels_csv).open(encoding="utf-8"))}

    ckpt_path = Path(args.checkpoint)
    ckpt_meta = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt_meta["model_state_dict"]
    dropout = float(ckpt_meta.get("fc_dropout") or infer_fc_dropout(state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_catdog(fc_dropout=dropout)
    load_checkpoint(model, ckpt_path)
    model.to(device).eval()
    _, eval_t = make_transforms(224)

    threshold = args.threshold
    if threshold < 0:
        val_csv = Path(args.internal_val_predictions)
        if not val_csv.exists():
            # fallback: use threshold that worked on internal test from test1 val split file - use 0.5
            threshold = 0.5
        else:
            threshold = tune_threshold_on_internal_val(val_csv)

    margins = (args.crop_top, args.crop_bottom, args.crop_left, args.crop_right)
    rows: List[Dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for img_path in sorted(Path(args.images_dir).glob("ext_*.jpg")):
        ext_id = img_path.stem
        if ext_id not in labels:
            continue
        src = source_by_ext.get(ext_id) if args.use_light_crop_from_source else None
        pil = load_pil_for_external(img_path, src, margins, args.prep)
        pc = predict_prob_cat(model, pil, eval_t, device, args.tta)
        pred = 1 if pc >= threshold else 0
        yt = labels[ext_id]
        row = {
            "ext_id": ext_id,
            "true_label_id": yt,
            "true_label_name": "cat" if yt == 1 else "dog",
            "pred_label_id": pred,
            "pred_label_name": "cat" if pred == 1 else "dog",
            "prob_cat": pc,
            "prob_dog": 1.0 - pc,
            "threshold_cat": threshold,
            "correct": bool(pred == yt),
            "prep": args.prep,
            "tta": args.tta,
            "light_crop": args.use_light_crop_from_source,
        }
        rows.append(row)
        y_true.append(yt)
        y_pred.append(pred)

    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    metrics = {
        "n": len(rows),
        "accuracy": acc,
        "threshold_cat": threshold,
        "prep": args.prep,
        "tta": args.tta,
        "light_crop_margins": margins,
        "note": "Threshold tuned on internal validation predictions only; external labels not used for tuning.",
    }
    if rows:
        yt_a = np.asarray(y_true)
        yp_a = np.asarray(y_pred)
        metrics["cat_recall"] = float(np.mean(yp_a[yt_a == 1] == 1)) if np.any(yt_a == 1) else None
        metrics["dog_recall"] = float(np.mean(yp_a[yt_a == 0] == 0)) if np.any(yt_a == 0) else None

    (out_dir / "external_predictions_adapted.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "external_metrics_adapted.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    for r in rows:
        if r["true_label_id"] == 1:
            print(r["ext_id"], "P(cat)=", f"{r['prob_cat']:.4f}", "pred", r["pred_label_name"], "OK" if r["correct"] else "MISS")


if __name__ == "__main__":
    main()
