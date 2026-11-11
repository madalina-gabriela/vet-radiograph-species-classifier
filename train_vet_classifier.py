#!/usr/bin/env python3
"""
Comparable training for ResNet-18, DenseNet-121, and EfficientNet-B0 on vet_dataset.

Protocol (architecture comparison):
  - Same patient split, preprocessing, augmentation, batch size, optimizer, loss
  - LR chosen from {1e-4, 3e-4} per architecture (separate runs; pick best on validation)
  - Max 50 epochs (5 classifier-only + fine-tune last block until cap or early stop)
  - Checkpoint = highest validation macro-F1
  - Early stopping: 10 epochs without validation macro-F1 improvement
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader

from architectures import build_architecture, count_trainable_parameters, set_training_phase
from experiment_registry import apply_experiment_to_namespace, get_experiment, list_experiments
from dataset_utils import (
    build_patient_records,
    compute_dataset_statistics,
    flatten_patient_records,
    load_split_from_json,
    patient_key_for_image,
    resolve_dicom_path,
    split_to_json_dict,
    splits_file_is_compatible,
    stratified_split_by_image,
    stratified_split_by_patient,
)
from dicom_preprocess import CACHE_VERSION, dicom_to_uint8
from eval_utils import aggregate_predictions_by_patient, patient_prediction_rows
from train_catdog import compute_metrics, predict, seed_everything, train_one_epoch
from transforms_vet import make_transforms
from vet_dataset import VetDicomDataset
from vet_training import evaluate_loader

SEED_DEFAULT = 42
PROTOCOL_VERSION = "compare_v1_macro_f1"
LR_CANDIDATES = (1e-4, 3e-4)


def run_training(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    seed_everything(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    splits_path = Path(args.splits_file)

    dataset_stats = compute_dataset_statistics(data_root, cats_dir=args.cats_dir, dogs_dir=args.dogs_dir)
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(dataset_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    patient_records = build_patient_records(data_root, cats_dir=args.cats_dir, dogs_dir=args.dogs_dir)
    samples = flatten_patient_records(patient_records)
    if not samples:
        raise RuntimeError("No DICOM samples found.")

    sample_paths = [p for p, _ in samples]
    if splits_file_is_compatible(splits_path, sample_paths, data_root, args.cats_dir, args.dogs_dir):
        splits = load_split_from_json(json.loads(splits_path.read_text(encoding="utf-8")))
        print(f"Loaded splits from {splits_path}")
    else:
        if args.split_level == "patient":
            splits = stratified_split_by_patient(
                records=patient_records,
                seed=args.seed,
                test_size=args.test_size,
                val_size=args.val_size,
            )
        else:
            splits = stratified_split_by_image(
                samples=samples,
                seed=args.seed,
                test_size=args.test_size,
                val_size=args.val_size,
            )
        splits_path.write_text(
            json.dumps(split_to_json_dict(splits, data_root), indent=2),
            encoding="utf-8",
        )

    def to_samples(raw: List) -> List:
        return [
            (resolve_dicom_path(str(p), data_root, args.cats_dir, args.dogs_dir), int(y)) for p, y in raw
        ]

    train_samples = to_samples(splits.train)
    val_samples = to_samples(splits.val)
    test_samples = to_samples(splits.test)

    train_t, eval_t = make_transforms(
        args.img_size,
        augmentation=args.augmentation,
        resize_mode=args.resize_mode,
        train_geom=args.train_geom,
    )
    train_ds = VetDicomDataset(
        train_samples, cache_dir, train_t, args.force_recache, offline_preprocess=args.offline_preprocess
    )
    val_ds = VetDicomDataset(
        val_samples, cache_dir, eval_t, args.force_recache, offline_preprocess=args.offline_preprocess
    )
    test_ds = VetDicomDataset(
        test_samples, cache_dir, eval_t, args.force_recache, offline_preprocess=args.offline_preprocess
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | Experiment: {args.experiment_code} | Arch: {args.architecture} | "
        f"LR: {args.lr} | pretrain={args.pretrained} | finetune_scope={args.finetune_scope} | "
        f"aug={args.augmentation} | offline={args.offline_preprocess} | resize={args.resize_mode}"
    )

    model = build_architecture(
        args.architecture,
        num_classes=2,
        fc_dropout=args.fc_dropout,
        pretrained=args.pretrained,
    )
    model = model.to(device)

    n_cat = sum(y for _, y in train_samples)
    n_dog = len(train_samples) - n_cat
    total = float(len(train_samples))
    class_weights = torch.tensor(
        [total / (2.0 * max(1.0, float(n_dog))), total / (2.0 * max(1.0, float(n_cat)))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    history: List[Dict[str, Any]] = []
    best_val_macro_f1 = -1.0
    best_val_acc = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    epochs_without_improve = 0
    global_epoch = 0
    phase = "head"

    set_training_phase(model, args.architecture, "head", finetune_scope=args.finetune_scope)
    n_trainable_head = count_trainable_parameters(model)
    print(f"Trainable parameters (head phase): {n_trainable_head:,}")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    while global_epoch < args.max_epochs:
        global_epoch += 1
        if global_epoch == args.epochs_head + 1 and phase == "head":
            phase = "finetune"
            set_training_phase(model, args.architecture, "finetune", finetune_scope=args.finetune_scope)
            print(f"Trainable parameters (finetune phase): {count_trainable_parameters(model):,}")
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            print(f"--- Switching to fine-tune (epoch {global_epoch}) ---")

        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
            max_grad_norm=args.max_grad_norm if phase == "finetune" else 0.0,
        )
        va_loss, va_acc, va_macro_f1 = evaluate_loader(model, val_loader, device, criterion)
        row = {
            "stage": phase,
            "epoch": global_epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_macro_f1": va_macro_f1,
        }
        history.append(row)
        print(
            f"[{phase}] epoch {global_epoch}/{args.max_epochs}: "
            f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} val_macro_f1={va_macro_f1:.4f}"
        )

        if va_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = va_macro_f1
            best_val_acc = va_acc
            best_epoch = global_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {global_epoch} (no val macro-F1 improvement for {args.early_stop_patience} epochs)")
                break

        if phase == "head" and global_epoch >= args.epochs_head:
            continue

    training_time_s = time.perf_counter() - t0

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")

    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state,
        "architecture": args.architecture,
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_val_acc": float(best_val_acc),
        "selected_epoch": int(best_epoch),
        "class_mapping": {"0": "dog", "1": "cat"},
        "img_size": int(args.img_size),
        "cache_version": CACHE_VERSION,
        "fc_dropout": float(args.fc_dropout),
        "lr": float(args.lr),
        "preprocessing": "modality_lut+voi_lut+percentile1-99+monochrome1_invert",
        "protocol_version": PROTOCOL_VERSION,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    torch.save(best_state, output_dir / "best_model_state.pt")

    y_true, y_pred, y_prob_cat = predict(model, test_loader, device, tta=args.tta)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_prob_cat=y_prob_cat)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    test_patient_keys = [patient_key_for_image(p, data_root) for p, _ in test_samples]
    y_true_p, y_pred_p, y_prob_cat_p = aggregate_predictions_by_patient(test_patient_keys, y_true, y_prob_cat)
    metrics_patient = compute_metrics(y_true=y_true_p, y_pred=y_pred_p, y_prob_cat=y_prob_cat_p)
    cm_patient = confusion_matrix(y_true_p, y_pred_p, labels=[0, 1])
    patient_rows = patient_prediction_rows(test_patient_keys, y_true, y_prob_cat, class_name={0: "dog", 1: "cat"})

    protocol = {
        "version": PROTOCOL_VERSION,
        "checkpoint_rule": "best validation macro-F1",
        "early_stop_patience": args.early_stop_patience,
        "early_stop_metric": "val_macro_f1",
        "max_epochs": args.max_epochs,
        "epochs_head_only": args.epochs_head,
        "finetune_scope": args.finetune_scope,
        "lr": args.lr,
        "lr_candidates_for_tuning": list(LR_CANDIDATES),
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "loss": "weighted CrossEntropyLoss",
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "tta_at_test": args.tta,
    }

    metrics_payload: Dict[str, Any] = {
        "experiment": {
            "code": args.experiment_code,
            "title": args.experiment_title,
            "description": get_experiment(args.experiment_code).get("description", ""),
        },
        "architecture": args.architecture,
        "pretrained": args.pretrained,
        "finetune_scope": args.finetune_scope,
        "augmentation": args.augmentation,
        "offline_preprocess": args.offline_preprocess,
        "resize_mode": args.resize_mode,
        "train_geom": args.train_geom,
        "protocol": protocol,
        "seed": args.seed,
        "data_root": str(data_root),
        "split_level": splits.split_level,
        "preprocessing": {
            "offline": "dicom_preprocess.dicom_to_uint8",
            "online_train": "make_transforms train_t",
            "online_eval": "make_transforms eval_t",
            "cache_version": CACHE_VERSION,
        },
        "training_time_seconds": round(training_time_s, 2),
        "selected_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_val_acc": float(best_val_acc),
        "metrics_image_level": metrics,
        "metrics_patient_level": metrics_patient,
        "metrics": metrics,
        "confusion_matrix_image_level": cm.tolist(),
        "confusion_matrix_patient_level": cm_patient.tolist(),
        "class_mapping": {"0": "dog", "1": "cat"},
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    _write_predictions(output_dir, test_samples, data_root, y_pred, y_prob_cat)
    with (output_dir / "predictions_patient_level.json").open("w", encoding="utf-8") as f:
        json.dump(patient_rows, f, indent=2)

    _plot_confusion(cm, output_dir / "confusion_matrix.png")
    try:
        fig = plt.figure(figsize=(5.0, 4.8))
        RocCurveDisplay.from_predictions(y_true, y_prob_cat, name="Cat vs Dog", ax=plt.gca())
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close(fig)
    except Exception:
        pass

    print(f"Training time: {training_time_s:.1f} s | Selected epoch: {best_epoch} | Best val macro-F1: {best_val_macro_f1:.4f}")
    print("Test accuracy (image):", metrics.get("accuracy"))


def _write_predictions(output_dir, test_samples, data_root, y_pred, y_prob_cat) -> None:
    class_name = {0: "dog", 1: "cat"}
    rows = []
    for i, ((img_path, true_label), pred_label, prob_cat) in enumerate(
        zip(test_samples, y_pred.tolist(), y_prob_cat.tolist())
    ):
        prob_cat_f = float(prob_cat)
        rows.append(
            {
                "index": i,
                "image_path": str(img_path),
                "patient_key": patient_key_for_image(img_path, data_root),
                "true_label_id": int(true_label),
                "true_label_name": class_name[int(true_label)],
                "pred_label_id": int(pred_label),
                "pred_label_name": class_name[int(pred_label)],
                "prob_dog": float(1.0 - prob_cat_f),
                "prob_cat": prob_cat_f,
                "correct": bool(int(true_label) == int(pred_label)),
            }
        )
    with (output_dir / "predictions_detailed.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (output_dir / "predictions_detailed.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_confusion(cm, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["dog (0)", "cat (1)"],
        yticklabels=["dog (0)", "cat (1)"],
        xlabel="Predicted",
        ylabel="True",
        title="Confusion matrix (test)",
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparable vet_dataset classifier training (M0–M8)")
    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list_experiments(),
        help="Experiment code: M0=ResNet baseline, M1=DenseNet, … run one at a time",
    )
    p.add_argument("--lr", type=float, required=True, help="Learning rate (1e-4 or 3e-4; run both via run_experiment.sh)")
    p.add_argument("--architecture", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--finetune_scope",
        type=str,
        default="last_block",
        choices=("last_block", "full"),
        help=argparse.SUPPRESS,
    )
    p.add_argument("--augmentation", type=str, default="default", choices=("default", "none"))
    p.add_argument("--offline_preprocess", type=str, default="default", choices=("default", "clahe"))
    p.add_argument("--resize_mode", type=str, default="center_crop", choices=("center_crop", "letterbox"))
    p.add_argument(
        "--train_geom",
        type=str,
        default="default",
        choices=("default", "affine_only", "with_rrc"),
        help=argparse.SUPPRESS,
    )
    p.add_argument("--data_root", type=str, default="/home/miorga/Doctorat/vet_dataset")
    p.add_argument("--cats_dir", type=str, default="Pisici")
    p.add_argument("--dogs_dir", type=str, default="Catei")
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/miorga/Doctorat/test_comparison/resnet18/lr_1e-4_seed_42",
    )
    p.add_argument("--cache_dir", type=str, default="/home/miorga/Doctorat/vet1_catdog/cache_dicom_uint8_vet_dataset_v2")
    p.add_argument(
        "--splits_file",
        type=str,
        default=str(Path(__file__).resolve().parent / "splits_vet_dataset_patient.json"),
    )
    p.add_argument("--split_level", type=str, choices=("patient", "image"), default="patient")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--epochs_head", type=int, default=5, help="Classifier-only epochs before last-block fine-tune")
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--fc_dropout", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no_tta", action="store_false", dest="tta")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--force_recache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_experiment_to_namespace(args)
    if args.lr not in LR_CANDIDATES:
        print(f"Warning: lr={args.lr} is outside the comparison grid {LR_CANDIDATES}")
    run_training(args)


if __name__ == "__main__":
    main()
