#!/usr/bin/env python3
"""
Few-step FC-only adaptation on external JPEG cohort (domain shift mitigation).

WARNING: This uses external labels during training — report separately from the
primary locked test evaluation. Suitable for pilot external usability, not unbiased external validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from external_validate_adapted import load_pil_for_external, predict_prob_cat, uint8_from_gray_array
from model_utils import build_resnet18_catdog, infer_fc_dropout, load_checkpoint
from train_catdog import make_transforms


class ExternalJPEGDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, Path | None, int]], margins, prep: str, eval_t) -> None:
        self.items = items
        self.margins = margins
        self.prep = prep
        self.eval_t = eval_t

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        anon_path, src_path, label = self.items[idx]
        pil = load_pil_for_external(anon_path, src_path, self.margins, self.prep)
        x = self.eval_t(pil)
        return x, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/miorga/Doctorat/vet1_catdog/results_vet_dataset/best_model.pt")
    parser.add_argument("--labels_csv", default="/home/miorga/Doctorat/vet_external_validation/labels.csv")
    parser.add_argument("--images_dir", default="/home/miorga/Doctorat/vet_external_validation/anonymized")
    parser.add_argument("--manifest_json", default="/home/miorga/Doctorat/vet_external_validation/anonymization_manifest.json")
    parser.add_argument("--output_dir", default="/home/miorga/Doctorat/vet_external_validation/results_fc_adapt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--prep", default="percentile")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    source_by_ext = {m["ext_id"]: Path("/home/miorga/Downloads/Fwd_ rx/test") / m["source_filename"] for m in manifest}
    labels = {r["ext_id"]: int(r["true_label_id"]) for r in csv.DictReader(Path(args.labels_csv).open(encoding="utf-8"))}

    items: List[Tuple[Path, Path | None, int]] = []
    for p in sorted(Path(args.images_dir).glob("ext_*.jpg")):
        eid = p.stem
        if eid in labels:
            items.append((p, source_by_ext.get(eid), labels[eid]))

    ckpt_path = Path(args.checkpoint)
    ckpt_meta = torch.load(str(ckpt_path), map_location="cpu")
    dropout = float(ckpt_meta.get("fc_dropout") or infer_fc_dropout(ckpt_meta["model_state_dict"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet18_catdog(fc_dropout=dropout)
    load_checkpoint(model, ckpt_path)
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

    _, eval_t = make_transforms(224)
    margins = (0.10, 0.16, 0.02, 0.10)
    ds = ExternalJPEGDataset(items, margins, args.prep, eval_t)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    n_cat = sum(y for _, _, y in items)
    n_dog = len(items) - n_cat
    w_dog = len(items) / (2.0 * max(1, n_dog))
    w_cat = len(items) / (2.0 * max(1, n_cat))
    weights = torch.tensor([w_dog, w_cat], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model.train()
    for ep in range(1, args.epochs + 1):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            total += float(loss.item())
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep} loss={total/max(1,len(loader)):.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "note": "FC adapted on external n=12"}, out_dir / "fc_adapted.pt")

    model.eval()
    rows = []
    correct = 0
    cat_hit = 0
    cat_n = 0
    for anon_path, src_path, yt in items:
        pil = load_pil_for_external(anon_path, src_path, margins, args.prep)
        pc = predict_prob_cat(model, pil, eval_t, device, tta=True)
        pred = 1 if pc >= 0.5 else 0
        ok = pred == yt
        correct += int(ok)
        if yt == 1:
            cat_n += 1
            cat_hit += int(pred == 1)
        rows.append({"ext_id": anon_path.stem, "true": yt, "pred": pred, "prob_cat": pc, "correct": ok})

    metrics = {
        "accuracy": correct / len(items),
        "cat_recall": cat_hit / max(1, cat_n),
        "dog_recall": 1.0 - (sum(1 for r in rows if r["true"] == 0 and r["pred"] == 1) / max(1, n_dog)),
        "n": len(items),
        "method": "FC-only adaptation on external pilot (uses external labels)",
    }
    (out_dir / "metrics_fc_adapt.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "predictions_fc_adapt.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
