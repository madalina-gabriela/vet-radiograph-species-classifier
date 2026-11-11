#!/usr/bin/env python3
"""Run trained ResNet-18 on anonymized external JPEG radiographs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from model_utils import build_resnet18_catdog, infer_fc_dropout, load_checkpoint
from train_catdog import make_transforms


def jpeg_to_pil_gray(path: Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("L")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/miorga/Doctorat/vet_external_validation/anonymized",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="/home/miorga/Doctorat/vet_external_validation/labels.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/miorga/Doctorat/vet1_catdog/results_vet_dataset/best_model.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/miorga/Doctorat/vet_external_validation/results",
    )
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_path = Path(args.labels_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_by_id: Dict[str, int] = {}
    if labels_path.exists():
        for row in csv.DictReader(labels_path.open(encoding="utf-8")):
            if row.get("true_label_id", "").strip() != "":
                label_by_id[row["ext_id"]] = int(row["true_label_id"])

    ckpt_path = Path(args.checkpoint)
    ckpt_meta = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt_meta["model_state_dict"] if isinstance(ckpt_meta, dict) else ckpt_meta
    dropout = float(ckpt_meta.get("fc_dropout") or infer_fc_dropout(state))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_catdog(fc_dropout=dropout)
    load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()

    _, eval_t = make_transforms(args.img_size)

    rows: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for img_path in sorted(images_dir.glob("ext_*.jpg")):
        ext_id = img_path.stem
        pil = jpeg_to_pil_gray(img_path)
        x = eval_t(pil).unsqueeze(0).to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
        pred = int(np.argmax(probs))
        prob_cat = float(probs[1])
        row: Dict[str, Any] = {
            "ext_id": ext_id,
            "image_path": str(img_path),
            "pred_label_id": pred,
            "pred_label_name": "cat" if pred == 1 else "dog",
            "prob_cat": prob_cat,
            "prob_dog": float(probs[0]),
        }
        if ext_id in label_by_id:
            yt = label_by_id[ext_id]
            row["true_label_id"] = yt
            row["true_label_name"] = "cat" if yt == 1 else "dog"
            row["correct"] = bool(yt == pred)
            y_true.append(yt)
            y_pred.append(pred)
        rows.append(row)

    (out_dir / "external_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "external_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metrics: Dict[str, Any] = {
        "n_images": len(rows),
        "n_labeled": len(y_true),
        "checkpoint": str(ckpt_path),
        "images_dir": str(images_dir),
        "note": "Pilot external set; JPEG thumbnails; second clinic; not used in training.",
    }

    if y_true:
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "precision_dog": float(prec[0]),
                "recall_dog": float(rec[0]),
                "f1_dog": float(f1[0]),
                "support_dog": int(sup[0]),
                "precision_cat": float(prec[1]),
                "recall_cat": float(rec[1]),
                "f1_cat": float(f1[1]),
                "support_cat": int(sup[1]),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            }
        )

    (out_dir / "external_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
