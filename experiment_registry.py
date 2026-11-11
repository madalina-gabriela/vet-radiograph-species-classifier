"""
Model comparison experiments M0–M10 (run one at a time).

Shared protocol (all M*): patient split, batch 16, img 224, AdamW, weighted CE + label smoothing,
max 50 epochs, 5 head-only then fine-tune, early stop 10 @ val macro-F1, checkpoint = best val macro-F1,
LR tuned on validation from {1e-4, 3e-4} per experiment.

finetune_scope:
  - last_block (default M0–M7): unfreeze last backbone block + head after epoch 5.
  - full (M8): unfreeze all layers after epoch 5 (scratch end-to-end control).
"""

from __future__ import annotations

from typing import Any, Dict, List

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "M0": {
        "code": "M0",
        "title": "ResNet-18 reference",
        "description": "Baseline performance",
        "architecture": "resnet18",
        "pretrained": True,
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "center_crop",
    },
    "M1": {
        "code": "M1",
        "title": "DenseNet-121",
        "description": "Architecture change",
        "architecture": "densenet121",
        "pretrained": True,
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "center_crop",
    },
    "M2": {
        "code": "M2",
        "title": "EfficientNet-B0",
        "description": "Performance vs compute",
        "architecture": "efficientnet_b0",
        "pretrained": True,
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "center_crop",
    },
    "M3": {
        "code": "M3",
        "title": "ResNet-18 + CBAM",
        "description": "Attention mechanism",
        "architecture": "resnet18_cbam",
        "pretrained": True,
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "center_crop",
    },
    "M4": {
        "code": "M4",
        "title": "ResNet-18 no ImageNet pretrain",
        "description": "Pretraining ablation (same partial fine-tune as M0; early blocks stay frozen at random init)",
        "architecture": "resnet18",
        "pretrained": False,
        "finetune_scope": "last_block",
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "center_crop",
    },
    "M5": {
        "code": "M5",
        "title": "ResNet-18 no augmentation",
        "description": "Augmentation ablation (train = deterministic resize/crop)",
        "architecture": "resnet18",
        "pretrained": True,
        "augmentation": "none",
        "offline_preprocess": "default",
        "resize_mode": "center_crop",
    },
    "M6": {
        "code": "M6",
        "title": "ResNet-18 + CLAHE",
        "description": "Local contrast enhancement (offline CLAHE after DICOM LUT)",
        "architecture": "resnet18",
        "pretrained": True,
        "augmentation": "default",
        "offline_preprocess": "clahe",
        "resize_mode": "center_crop",
    },
    "M7": {
        "code": "M7",
        "title": "ResNet-18 letterbox resize",
        "description": "Eval: letterbox scale; train: flip/affine (no RRC). Confounded vs M0 (eval crop + RRC). On vet_dataset all uint8 are 2880² → no padding, only downscale vs center-crop eval.",
        "architecture": "resnet18",
        "pretrained": True,
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "letterbox",
        "train_geom": "default",
    },
    "M9": {
        "code": "M9",
        "title": "ResNet-18 center-crop eval, affine train (no RRC)",
        "description": "Same eval as M0; train geom matched to M7 (Resize 224 + flip/affine, no RandomResizedCrop)",
        "architecture": "resnet18",
        "pretrained": True,
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "center_crop",
        "train_geom": "affine_only",
    },
    "M10": {
        "code": "M10",
        "title": "ResNet-18 letterbox + RandomResizedCrop train",
        "description": "Same eval as M7; train adds RRC after letterbox (geom matched toward M0)",
        "architecture": "resnet18",
        "pretrained": True,
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "letterbox",
        "train_geom": "with_rrc",
    },
    "M8": {
        "code": "M8",
        "title": "ResNet-18 scratch, full-network fine-tune",
        "description": "No ImageNet; after 5 head epochs all layers trainable (fair scratch vs M0/M4)",
        "architecture": "resnet18",
        "pretrained": False,
        "finetune_scope": "full",
        "augmentation": "default",
        "offline_preprocess": "default",
        "resize_mode": "center_crop",
    },
}


def list_experiments() -> List[str]:
    return sorted(EXPERIMENTS.keys())


def get_experiment(code: str) -> Dict[str, Any]:
    key = code.strip().upper()
    if key not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {code!r}. Choose from: {', '.join(list_experiments())}")
    return dict(EXPERIMENTS[key])


def apply_experiment_to_namespace(args: Any) -> None:
    """Mutate argparse Namespace from --experiment code."""
    exp = get_experiment(args.experiment)
    args.experiment_code = exp["code"]
    args.experiment_title = exp["title"]
    args.architecture = exp["architecture"]
    args.pretrained = bool(exp["pretrained"])
    args.finetune_scope = str(exp.get("finetune_scope", "last_block"))
    args.augmentation = exp["augmentation"]
    args.offline_preprocess = exp["offline_preprocess"]
    args.resize_mode = exp["resize_mode"]
    args.train_geom = str(exp.get("train_geom", "default"))
