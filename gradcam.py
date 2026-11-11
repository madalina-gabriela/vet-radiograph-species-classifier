#!/usr/bin/env python3
"""
Grad-CAM for vet compare_v1 classifiers (ResNet-18, CBAM, DenseNet-121, EfficientNet-B0).

Generates heatmaps for the predicted class (or the ground-truth class) to inspect
whether saliency aligns with class-relevant anatomy.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from architectures import build_architecture
from dicom_preprocess import dicom_to_uint8
from model_utils import infer_fc_dropout
from transforms_vet import make_transforms


@dataclass
class GradCamResult:
    image_path: Path
    true_label: int
    pred_label: int
    prob_cat: float
    heatmap: np.ndarray
    overlay: np.ndarray
    base_uint8: np.ndarray


def get_gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    arch = architecture.lower().replace("-", "_")
    if arch in ("resnet18", "resnet18_cbam"):
        return model.layer4[-1]
    if arch == "densenet121":
        return model.features.norm5
    if arch == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target registered for architecture {architecture!r}")


def load_model_from_checkpoint(ckpt_path: Path) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt_meta = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt_meta, dict) or "model_state_dict" not in ckpt_meta:
        raise ValueError(f"Expected compare_v1 checkpoint dict at {ckpt_path}")
    state = ckpt_meta["model_state_dict"]
    architecture = str(ckpt_meta.get("architecture", "resnet18"))
    dropout = float(ckpt_meta.get("fc_dropout") or infer_fc_dropout(state))
    model = build_architecture(architecture, fc_dropout=dropout, pretrained=False)
    model.load_state_dict(state)
    return model, ckpt_meta


class GradCAM:
    """Grad-CAM on the last spatial feature block of the backbone."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def _save_activation(self, _module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module: nn.Module, _grad_in: Any, grad_out: Any) -> None:
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_index: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_index is None:
            class_index = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_index]
        score.backward()

        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam.astype(np.float32)


def overlay_heatmap(base_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = base_uint8.shape
    cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize((w, h), Image.BILINEAR)
    cam_u8 = np.asarray(cam_img, dtype=np.float32) / 255.0
    cmap = plt.cm.jet(cam_u8)[..., :3]
    base_rgb = np.stack([base_uint8, base_uint8, base_uint8], axis=-1).astype(np.float32) / 255.0
    blended = (1 - alpha) * base_rgb + alpha * cmap
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def run_gradcam_on_path(
    model: nn.Module,
    grad_cam: GradCAM,
    transform: transforms.Compose,
    image_path: Path,
    device: torch.device,
    true_label: int,
    pred_label: int,
    prob_cat: float,
    cam_class_index: Optional[int] = None,
) -> GradCamResult:
    base = dicom_to_uint8(image_path)
    pil = Image.fromarray(base, mode="L")
    x = transform(pil).unsqueeze(0).to(device)
    x.requires_grad_(True)

    model.eval()
    cam = grad_cam(x, class_index=cam_class_index if cam_class_index is not None else pred_label)
    cam_resized = (
        np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize(
                (base.shape[1], base.shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    overlay = overlay_heatmap(base, cam_resized)

    return GradCamResult(
        image_path=image_path,
        true_label=true_label,
        pred_label=pred_label,
        prob_cat=prob_cat,
        heatmap=cam_resized,
        overlay=overlay,
        base_uint8=base,
    )


def save_panel(results: List[GradCamResult], out_path: Path, title: str) -> None:
    n = len(results)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.asarray(axes).reshape(1, -1)
    elif cols == 1:
        axes = np.asarray(axes).reshape(-1, 1)

    names = {0: "dog", 1: "cat"}
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        r = results[idx]
        ax.imshow(r.overlay)
        ax.set_title(
            f"T:{names[r.true_label]} P:{names[r.pred_label]}\nP(cat)={r.prob_cat:.2f}",
            fontsize=9,
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pick_examples(csv_path: Path, per_category: int) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    buckets: Dict[str, List[Dict[str, str]]] = {
        "correct_dog": [],
        "correct_cat": [],
        "wrong_dog_as_cat": [],
        "wrong_cat_as_dog": [],
    }
    for r in rows:
        t, p = int(r["true_label_id"]), int(r["pred_label_id"])
        correct = r["correct"] in ("True", "true", "1")
        prob_cat = float(r["prob_cat"])
        if correct and t == 0:
            buckets["correct_dog"].append(r)
        elif correct and t == 1:
            buckets["correct_cat"].append(r)
        elif not correct and t == 0 and p == 1:
            buckets["wrong_dog_as_cat"].append(r)
        elif not correct and t == 1 and p == 0:
            buckets["wrong_cat_as_dog"].append(r)

    def sort_conf(rows_in: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return sorted(rows_in, key=lambda x: max(float(x["prob_cat"]), float(x["prob_dog"])), reverse=True)

    picked: List[Dict[str, str]] = []
    for key in ("correct_dog", "correct_cat", "wrong_dog_as_cat", "wrong_cat_as_dog"):
        picked.extend(sort_conf(buckets[key])[:per_category])
    return picked


def infer_resize_mode(ckpt_meta: Dict[str, Any], metrics_path: Optional[Path]) -> str:
    if metrics_path and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("resize_mode"):
            return str(metrics["resize_mode"])
    prep = ckpt_meta.get("preprocessing") or {}
    if isinstance(prep, dict) and prep.get("resize_mode"):
        return str(prep["resize_mode"])
    return "center_crop"


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM explainability for cat/dog radiographs.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/miorga/Doctorat/vet1_catdog/results_vet_dataset/best_model.pt",
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default="/home/miorga/Doctorat/test1/predictions_detailed.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/miorga/Doctorat/test1/gradcam",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="Optional metrics.json (resize_mode, experiment code for titles).",
    )
    parser.add_argument("--img_size", type=int, default=0, help="0 = read from checkpoint.")
    parser.add_argument("--resize_mode", type=str, default="", choices=["", "center_crop", "letterbox"])
    parser.add_argument(
        "--cam_class",
        type=str,
        default="predicted",
        choices=["predicted", "true"],
        help="Backprop target class for Grad-CAM.",
    )
    parser.add_argument("--per_category", type=int, default=3)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt_meta = load_model_from_checkpoint(ckpt_path)
    architecture = str(ckpt_meta.get("architecture", "resnet18"))
    img_size = int(args.img_size or ckpt_meta.get("img_size") or 224)
    metrics_path = Path(args.metrics_json) if args.metrics_json else None
    resize_mode = args.resize_mode or infer_resize_mode(ckpt_meta, metrics_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    target_layer = get_gradcam_target_layer(model, architecture)
    grad_cam = GradCAM(model, target_layer)
    transform, _ = make_transforms(img_size=img_size, augmentation="none", resize_mode=resize_mode)

    experiment_code = ""
    if metrics_path and metrics_path.is_file():
        experiment_code = json.loads(metrics_path.read_text(encoding="utf-8")).get("experiment", {}).get("code", "")

    examples = pick_examples(Path(args.predictions_csv), args.per_category)
    results: List[GradCamResult] = []
    manifest: List[Dict[str, Any]] = []

    for row in examples:
        path = Path(row["image_path"])
        true_label = int(row["true_label_id"])
        pred_label = int(row["pred_label_id"])
        cam_label = true_label if args.cam_class == "true" else pred_label
        res = run_gradcam_on_path(
            model,
            grad_cam,
            transform,
            path,
            device,
            true_label=true_label,
            pred_label=pred_label,
            prob_cat=float(row["prob_cat"]),
            cam_class_index=cam_label,
        )
        results.append(res)
        stem = path.parent.name.replace(" ", "_") + "_" + path.name[:8]
        single_out = out_dir / f"gradcam_{stem}.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(res.base_uint8, cmap="gray")
        axes[0].set_title("DICOM (preprocessed)")
        axes[0].axis("off")
        axes[1].imshow(res.heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        axes[2].imshow(res.overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(single_out, dpi=180)
        plt.close(fig)
        manifest.append(
            {
                "image_path": str(path),
                "true_label_id": true_label,
                "pred_label_id": pred_label,
                "gradcam_class_id": cam_label,
                "prob_cat": res.prob_cat,
                "figure": str(single_out),
            }
        )

    title = "Grad-CAM (M2 EfficientNet-B0)" if experiment_code == "M2" else "Grad-CAM: representative test cases"
    if experiment_code and experiment_code != "M2":
        title = f"Grad-CAM ({experiment_code}, {architecture})"
    save_panel(results, out_dir / "gradcam_panel_manuscript.png", title)
    (out_dir / "gradcam_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    grad_cam.close()
    print(f"Saved {len(results)} Grad-CAM examples to {out_dir}")


if __name__ == "__main__":
    main()
