import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
from model_utils import build_densenet121_catdog


SEED_DEFAULT = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DicomCatDogDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        cache_dir: Path,
        transform: Optional[Any] = None,
        force_recache: bool = False,
    ) -> None:
        self.samples = samples
        self.cache_dir = cache_dir
        self.transform = transform
        self.force_recache = force_recache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, dcm_path: Path) -> Path:
        h = hashlib.sha1(f"{CACHE_VERSION}:{dcm_path}".encode("utf-8")).hexdigest()  # nosec - deterministic cache key
        return self.cache_dir / f"{h}.npy"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        dcm_path, label = self.samples[idx]
        cache_path = self._cache_path(dcm_path)

        if (not self.force_recache) and cache_path.exists():
            arr_uint8 = np.load(str(cache_path))
        else:
            arr_uint8 = dicom_to_uint8(dcm_path)
            np.save(str(cache_path), arr_uint8)

        # Keep single-channel grayscale; we'll adapt the network's first conv layer accordingly.
        img = Image.fromarray(arr_uint8, mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_cat: np.ndarray,
) -> Dict[str, Any]:
    # Binary: pos class = 1 (cat)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    # precision/recall/f1 are per-class in the same order as labels=[0,1]
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_dog": float(precision[0]),
        "recall_dog": float(recall[0]),
        "f1_dog": float(f1[0]),
        "support_dog": int(support[0]),
        "precision_cat": float(precision[1]),
        "recall_cat": float(recall[1]),
        "f1_cat": float(f1[1]),
        "support_cat": int(support[1]),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }

    # ROC-AUC if probabilities are usable (binary)
    try:
        auc = float(roc_auc_score(y_true, y_prob_cat))
        metrics["roc_auc_cat_vs_dog"] = auc
    except Exception:
        metrics["roc_auc_cat_vs_dog"] = None

    return metrics


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_prob_cat_all: List[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        if tta:
            x_flip = torch.flip(x, dims=[-1])
            probs_flip = torch.softmax(model(x_flip), dim=1)
            probs = (probs + probs_flip) * 0.5
        pred = torch.argmax(probs, dim=1)
        y_true_all.extend(y.detach().cpu().numpy().astype(int).tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().astype(int).tolist())
        y_prob_cat_all.extend(probs[:, 1].detach().cpu().numpy().astype(float).tolist())

    return (
        np.asarray(y_true_all, dtype=int),
        np.asarray(y_pred_all, dtype=int),
        np.asarray(y_prob_cat_all, dtype=np.float32),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_grad_norm: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().detach().cpu().item())
        n += int(y.size(0))

    return total_loss / max(1, n), correct / max(1, n)


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.detach().cpu().item()) * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().detach().cpu().item())
        n += int(y.size(0))

    return total_loss / max(1, n), correct / max(1, n)


def freeze_backbone(model: nn.Module) -> None:
    """Train classifier only (DenseNet-121)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier.")


def unfreeze_last_block_and_fc(model: nn.Module) -> None:
    # Last dense block + final norm + classifier (analogous to ResNet layer4 + fc).
    for name, param in model.named_parameters():
        if (
            name.startswith("classifier.")
            or name.startswith("features.denseblock4.")
            or name.startswith("features.norm5.")
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False


def make_transforms(img_size: int) -> Tuple[Any, Any]:
    # ImageNet mean/std averaged across RGB channels, used for a 1-channel adaptation.
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]
    mean = [float(np.mean(mean_rgb))]
    std = [float(np.mean(std_rgb))]

    train_t = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (img_size, img_size),
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.06, 0.06),
                scale=(0.92, 1.08),
                shear=4,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.15,
            ),
            transforms.RandomAutocontrast(p=0.25),
            transforms.ToTensor(),
            transforms.RandomApply(
                [transforms.Lambda(lambda t: t + torch.randn_like(t) * 0.02)],
                p=0.2,
            ),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_t = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_t, eval_t


def build_classifier_head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    return nn.Linear(in_features, num_classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/miorga/Doctorat/vet_dataset")
    parser.add_argument("--cats_dir", type=str, default="Pisici")
    parser.add_argument("--dogs_dir", type=str, default="Catei")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/miorga/Doctorat/test_densenet121/run_seed_42",
    )
    parser.add_argument("--cache_dir", type=str, default="/home/miorga/Doctorat/vet1_catdog/cache_dicom_uint8_vet_dataset_v2")
    parser.add_argument(
        "--splits_file",
        type=str,
        default=str(Path(__file__).resolve().parent / "splits_vet_dataset_patient.json"),
    )
    parser.add_argument(
        "--split_level",
        type=str,
        choices=("patient", "image"),
        default="patient",
        help="Split train/val/test by patient folder (recommended) or by individual DICOM.",
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_fc_only", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=8)
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Finetune epochs without val improvement before stop.")
    parser.add_argument("--lr_fc", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--fc_dropout", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--tta", action="store_true", default=True, help="Test-time horizontal flip averaging.")
    parser.add_argument("--no_tta", action="store_false", dest="tta")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument("--save_gallery_count", type=int, default=12)
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="After test evaluation, run Grad-CAM on representative test cases.",
    )
    parser.add_argument("--gradcam_per_category", type=int, default=2)
    args = parser.parse_args()

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
    print("Full dataset (before split):")
    print(f"  dogs: {dataset_stats['by_class']['dog']}")
    print(f"  cats: {dataset_stats['by_class']['cat']}")
    print(f"  total DICOM images: {dataset_stats['total_dicom_images']}")

    patient_records = build_patient_records(data_root, cats_dir=args.cats_dir, dogs_dir=args.dogs_dir)
    samples = flatten_patient_records(patient_records)
    if len(samples) == 0:
        raise RuntimeError("No DICOM samples found under patient folders.")

    sample_paths = [p for p, _ in samples]
    if splits_file_is_compatible(splits_path, sample_paths, data_root, args.cats_dir, args.dogs_dir):
        splits = load_split_from_json(json.loads(splits_path.read_text(encoding="utf-8")))
        print(f"Loaded splits from {splits_path} (level={splits.split_level})")
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
        print(f"Wrote new {args.split_level}-level splits to {splits_path}")

    def to_samples(raw: List[Tuple[str, int]]) -> List[Tuple[Path, int]]:
        return [
            (resolve_dicom_path(str(p), data_root, args.cats_dir, args.dogs_dir), int(y)) for p, y in raw
        ]

    train_samples = to_samples(splits.train)
    val_samples = to_samples(splits.val)
    test_samples = to_samples(splits.test)

    # Compute distribution for logging
    def dist(samps: List[Tuple[Path, int]]) -> Dict[str, int]:
        n_cat = sum(y for _, y in samps)
        n_dog = len(samps) - n_cat
        return {"cat": int(n_cat), "dog": int(n_dog), "total": int(len(samps))}

    def dist_patients(patient_lists: Dict[str, List[str]] | None, split_name: str) -> Dict[str, int]:
        if not patient_lists or split_name not in patient_lists:
            keys = {patient_key_for_image(p, data_root) for p, _ in {"train": train_samples, "val": val_samples, "test": test_samples}[split_name]}
        else:
            keys = set(patient_lists[split_name])
        n_cat = sum(1 for k in keys if k.startswith(f"{args.cats_dir}/"))
        n_dog = len(keys) - n_cat
        return {"cat": n_cat, "dog": n_dog, "total": len(keys)}

    print(f"Dataset split (images), level={splits.split_level}:")
    print("  train:", dist(train_samples), "| patients:", dist_patients(splits.patients, "train"))
    print("  val:", dist(val_samples), "| patients:", dist_patients(splits.patients, "val"))
    print("  test:", dist(test_samples), "| patients:", dist_patients(splits.patients, "test"))

    train_t, eval_t = make_transforms(args.img_size)

    train_ds = DicomCatDogDataset(train_samples, cache_dir=cache_dir, transform=train_t, force_recache=args.force_recache)
    val_ds = DicomCatDogDataset(val_samples, cache_dir=cache_dir, transform=eval_t, force_recache=args.force_recache)
    test_ds = DicomCatDogDataset(test_samples, cache_dir=cache_dir, transform=eval_t, force_recache=args.force_recache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = build_densenet121_catdog(num_classes=2, fc_dropout=args.fc_dropout)
    model = model.to(device)
    # Class imbalance: in this dataset dogs (0) are more frequent than cats (1).
    # Use inverse-frequency weights so the classifier doesn't overfit to the majority class.
    n_cat = sum(y for _, y in train_samples)
    n_dog = len(train_samples) - n_cat
    total = float(len(train_samples))
    # weight for each class: total / (2 * count)
    w_dog = total / (2.0 * max(1.0, float(n_dog)))
    w_cat = total / (2.0 * max(1.0, float(n_cat)))
    class_weights = torch.tensor([w_dog, w_cat], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # Stage 1: train only FC layer
    freeze_backbone(model)
    optimizer_fc = AdamW(model.parameters(), lr=args.lr_fc, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    history: List[Dict[str, Any]] = []
    best_val_acc = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs_fc_only + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer_fc, criterion)
        va_loss, va_acc = evaluate_one_epoch(model, val_loader, device, criterion)
        row = {"stage": "fc_only", "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        history.append(row)
        print(f"[FC] epoch {epoch}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Stage 2: unfreeze last block + FC
    unfreeze_last_block_and_fc(model)
    optimizer_ft = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_finetune, weight_decay=args.weight_decay)

    finetune_no_improve = 0

    for epoch in range(1, args.epochs_finetune + 1):
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer_ft,
            criterion,
            max_grad_norm=args.max_grad_norm,
        )
        va_loss, va_acc = evaluate_one_epoch(model, val_loader, device, criterion)
        row = {"stage": "finetune", "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        history.append(row)
        print(f"[FT] epoch {epoch}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            finetune_no_improve = 0
        else:
            finetune_no_improve += 1
            if finetune_no_improve >= args.early_stop_patience:
                print(f"[FT] early stopping (patience={args.early_stop_patience}) at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / "best_model_state.pt")
        checkpoint = {
            "model_state_dict": best_state,
            "best_val_acc": float(best_val_acc),
            "class_mapping": {"0": "dog", "1": "cat"},
            "img_size": int(args.img_size),
            "cache_version": CACHE_VERSION,
            "fc_dropout": float(args.fc_dropout),
            "preprocessing": "modality_lut+voi_lut+percentile1-99+monochrome1_invert",
            "architecture": "densenet121",
        }
        torch.save(checkpoint, output_dir / "best_model.pt")

    # Final test evaluation (image-level)
    y_true, y_pred, y_prob_cat = predict(model, test_loader, device, tta=args.tta)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_prob_cat=y_prob_cat)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    test_patient_keys = [patient_key_for_image(p, data_root) for p, _ in test_samples]
    y_true_p, y_pred_p, y_prob_cat_p = aggregate_predictions_by_patient(test_patient_keys, y_true, y_prob_cat)
    metrics_patient = compute_metrics(y_true=y_true_p, y_pred=y_pred_p, y_prob_cat=y_prob_cat_p)
    cm_patient = confusion_matrix(y_true_p, y_pred_p, labels=[0, 1])
    patient_rows = patient_prediction_rows(test_patient_keys, y_true, y_prob_cat, class_name={0: "dog", 1: "cat"})

    metrics_payload: Dict[str, Any] = {
        "architecture": "densenet121",
        "seed": args.seed,
        "data_root": str(data_root),
        "cats_dir": args.cats_dir,
        "dogs_dir": args.dogs_dir,
        "split_level": splits.split_level,
        "preprocessing": {
            "offline": "dicom_preprocess.dicom_to_uint8 (LUT + percentile + MONOCHROME1)",
            "online_train": "RandomResizedCrop, flip, affine, blur, autocontrast, light noise",
            "online_eval": "resize + center crop + normalize",
            "cache_version": CACHE_VERSION,
        },
        "training": {
            "epochs_fc_only": args.epochs_fc_only,
            "epochs_finetune_max": args.epochs_finetune,
            "early_stop_patience": args.early_stop_patience,
            "label_smoothing": args.label_smoothing,
            "fc_dropout": args.fc_dropout,
            "weight_decay": args.weight_decay,
            "tta": args.tta,
        },
        "dataset_statistics": dataset_stats,
        "split_patients": splits.patients,
        "splits": {"test_size": args.test_size, "val_size": args.val_size, "img_size": args.img_size, "batch_size": args.batch_size},
        "best_val_acc": float(best_val_acc),
        "metrics_image_level": metrics,
        "metrics_patient_level": metrics_patient,
        "metrics": metrics,
        "confusion_matrix_image_level": cm.tolist(),
        "confusion_matrix_patient_level": cm_patient.tolist(),
        "confusion_matrix": cm.tolist(),
        "class_mapping": {"0": "dog", "1": "cat"},
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    with (output_dir / "predictions_patient_level.json").open("w", encoding="utf-8") as f:
        json.dump(patient_rows, f, indent=2)

    # Detailed per-image predictions for auditability.
    class_name = {0: "dog", 1: "cat"}
    prediction_rows: List[Dict[str, Any]] = []
    for i, ((img_path, true_label), pred_label, prob_cat) in enumerate(zip(test_samples, y_pred.tolist(), y_prob_cat.tolist())):
        prob_cat_f = float(prob_cat)
        prob_dog_f = float(1.0 - prob_cat_f)
        row = {
            "index": i,
            "image_path": str(img_path),
            "patient_key": patient_key_for_image(img_path, data_root),
            "true_label_id": int(true_label),
            "true_label_name": class_name[int(true_label)],
            "pred_label_id": int(pred_label),
            "pred_label_name": class_name[int(pred_label)],
            "prob_dog": prob_dog_f,
            "prob_cat": prob_cat_f,
            "correct": bool(int(true_label) == int(pred_label)),
        }
        prediction_rows.append(row)

    with (output_dir / "predictions_detailed.json").open("w", encoding="utf-8") as f:
        json.dump(prediction_rows, f, indent=2)

    with (output_dir / "predictions_detailed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "image_path",
                "patient_key",
                "true_label_id",
                "true_label_name",
                "pred_label_id",
                "pred_label_name",
                "prob_dog",
                "prob_cat",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    def save_gallery(rows: List[Dict[str, Any]], out_path: Path, title: str) -> None:
        if len(rows) == 0:
            return
        n = min(args.save_gallery_count, len(rows))
        rows = rows[:n]
        cols = 4
        rws = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(4 * cols, 3.3 * rws))
        if rws == 1:
            axes = np.asarray(axes).reshape(1, -1)
        for idx in range(rws * cols):
            ax = axes[idx // cols, idx % cols]
            ax.axis("off")
            if idx >= n:
                continue
            row = rows[idx]
            img = dicom_to_uint8(Path(row["image_path"]))
            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"T:{row['true_label_name']} P:{row['pred_label_name']}\n"
                f"cat={row['prob_cat']:.2f} dog={row['prob_dog']:.2f}",
                fontsize=9,
            )
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    wrong_rows = [r for r in prediction_rows if not r["correct"]]
    wrong_rows = sorted(wrong_rows, key=lambda r: max(r["prob_cat"], r["prob_dog"]), reverse=True)
    right_rows = [r for r in prediction_rows if r["correct"]]
    right_rows = sorted(right_rows, key=lambda r: max(r["prob_cat"], r["prob_dog"]), reverse=True)

    save_gallery(
        wrong_rows,
        output_dir / "predictions_wrong_gallery.png",
        title="Most confident wrong predictions",
    )
    save_gallery(
        right_rows,
        output_dir / "predictions_correct_gallery.png",
        title="Most confident correct predictions",
    )

    # Confusion matrix plot
    fig = plt.figure(figsize=(5.0, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["dog (0)", "cat (1)"],
        yticklabels=["dog (0)", "cat (1)"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Cat vs Dog)",
    )
    # Annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # Print a compact summary
    print("Test metrics (image-level):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("Test metrics (patient-level, mean prob):")
    for k, v in metrics_patient.items():
        print(f"  {k}: {v}")
    print("Confusion matrix image [[dog->dog, dog->cat],[cat->dog, cat->cat]]:")
    print(cm)
    print("Confusion matrix patient:")
    print(cm_patient)

    # Simple ROC curve
    try:
        from sklearn.metrics import RocCurveDisplay

        fig = plt.figure(figsize=(5.0, 4.8))
        RocCurveDisplay.from_predictions(y_true, y_prob_cat, name="Cat vs Dog (pos=cat)", ax=plt.gca())
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close(fig)
    except Exception:
        pass

    if args.gradcam:
        print("Grad-CAM for DenseNet-121 is not wired in gradcam.py yet; skipping.")


if __name__ == "__main__":
    main()

