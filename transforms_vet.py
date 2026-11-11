"""Online transforms for vet experiments (augmentation / resize ablations)."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _norm_constants() -> Tuple[list, list]:
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]
    return [float(np.mean(mean_rgb))], [float(np.mean(std_rgb))]


class LetterboxToSquare:
    """Scale so the full image fits inside size×size, pad with min intensity."""

    def __init__(self, size: int, fill: int = 0) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("L", (self.size, self.size), color=self.fill)
        left = (self.size - nw) // 2
        top = (self.size - nh) // 2
        canvas.paste(img, (left, top))
        return canvas


def _eval_center_crop(img_size: int) -> transforms.Compose:
    mean, std = _norm_constants()
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _eval_letterbox(img_size: int) -> transforms.Compose:
    mean, std = _norm_constants()
    return transforms.Compose(
        [
            LetterboxToSquare(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_transforms(
    img_size: int = 224,
    augmentation: str = "default",
    resize_mode: str = "center_crop",
    train_geom: str = "default",
) -> Tuple[Any, Any]:
    mean, std = _norm_constants()
    if resize_mode == "letterbox":
        eval_t = _eval_letterbox(img_size)
    else:
        eval_t = _eval_center_crop(img_size)

    if augmentation == "none":
        return eval_t, eval_t

    train_steps = []
    use_rrc = train_geom == "with_rrc" or (train_geom == "default" and resize_mode == "center_crop")
    use_affine_only = train_geom == "affine_only" or (train_geom == "default" and resize_mode == "letterbox")

    if use_rrc and use_affine_only:
        raise ValueError("train_geom cannot combine RRC and affine_only")

    if resize_mode == "letterbox":
        train_steps.append(LetterboxToSquare(img_size))
    elif use_affine_only:
        # Match letterbox train path on square sources: full-field scale to img_size (no eval-style crop).
        train_steps.append(transforms.Resize((img_size, img_size)))
    else:
        pass  # RRC below produces img_size×img_size

    if use_rrc:
        train_steps.extend(
            [
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08), shear=4),
            ]
        )
    elif use_affine_only:
        train_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08), shear=4),
            ]
        )
    else:
        raise ValueError(f"Unknown train_geom {train_geom!r}")
    train_steps.extend(
        [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
            transforms.RandomAutocontrast(p=0.25),
            transforms.ToTensor(),
            transforms.RandomApply([transforms.Lambda(lambda t: t + torch.randn_like(t) * 0.02)], p=0.2),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_t = transforms.Compose(train_steps)
    return train_t, eval_t
