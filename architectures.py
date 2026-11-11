"""Architecture registry for comparable vet_dataset classifiers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, EfficientNet_B0_Weights, ResNet18_Weights

from model_utils import (
    _adapt_grayscale_conv,
    build_classifier_head,
    build_densenet121_catdog,
)
from resnet_cbam import ResNet18CBAM


def _build_resnet18(num_classes: int, fc_dropout: float, pretrained: bool) -> nn.Module:
    from torchvision import models

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = build_classifier_head(model.fc.in_features, num_classes, fc_dropout)
    model.conv1 = _adapt_grayscale_conv(model.conv1)
    return model


def _build_resnet18_cbam(num_classes: int, fc_dropout: float, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    return ResNet18CBAM(num_classes, fc_dropout, weights=weights)


def _build_efficientnet_b0(num_classes: int, fc_dropout: float, pretrained: bool) -> nn.Module:
    from torchvision import models

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = build_classifier_head(in_features, num_classes, fc_dropout)
    model.features[0][0] = _adapt_grayscale_conv(model.features[0][0])
    return model


def _freeze_resnet18_classifier_only(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def _unfreeze_resnet18_last_block(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4.") or name.startswith("fc.")


def _freeze_resnet18_cbam_classifier_only(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def _unfreeze_resnet18_cbam_last_block(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("fc.")
            or name.startswith("layer4.")
            or name.startswith("cbam.")
        )


def _freeze_densenet_classifier_only(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier.")


def _unfreeze_densenet_last_block(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("classifier.")
            or name.startswith("features.denseblock4.")
            or name.startswith("features.norm5.")
        )


def _freeze_efficientnet_classifier_only(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier.")


def _unfreeze_efficientnet_last_block(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("classifier.")
            or name.startswith("features.6.")
            or name.startswith("features.7.")
            or name.startswith("features.8.")
        )


def _wrap_densenet(pretrained: bool):
    def _fn(nc: int, drop: float) -> nn.Module:
        w = DenseNet121_Weights.DEFAULT if pretrained else None
        return build_densenet121_catdog(nc, drop, weights=w)

    return _fn


ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "build": lambda nc, drop, pre: _build_resnet18(nc, drop, pre),
        "freeze_head": _freeze_resnet18_classifier_only,
        "unfreeze_finetune": _unfreeze_resnet18_last_block,
    },
    "resnet18_cbam": {
        "build": lambda nc, drop, pre: _build_resnet18_cbam(nc, drop, pre),
        "freeze_head": _freeze_resnet18_cbam_classifier_only,
        "unfreeze_finetune": _unfreeze_resnet18_cbam_last_block,
    },
    "densenet121": {
        "build": lambda nc, drop, pre: _wrap_densenet(pre)(nc, drop),
        "freeze_head": _freeze_densenet_classifier_only,
        "unfreeze_finetune": _unfreeze_densenet_last_block,
    },
    "efficientnet_b0": {
        "build": lambda nc, drop, pre: _build_efficientnet_b0(nc, drop, pre),
        "freeze_head": _freeze_efficientnet_classifier_only,
        "unfreeze_finetune": _unfreeze_efficientnet_last_block,
    },
}


def list_architectures() -> List[str]:
    return sorted(ARCHITECTURES.keys())


def build_architecture(
    name: str,
    num_classes: int = 2,
    fc_dropout: float = 0.0,
    pretrained: bool = True,
) -> nn.Module:
    key = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "resnet18": "resnet18",
        "resnet18cbam": "resnet18_cbam",
        "densenet121": "densenet121",
        "efficientnetb0": "efficientnet_b0",
        "efficientnet_b0": "efficientnet_b0",
    }
    canon = aliases.get(key, name)
    if canon not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}. Choose from: {list_architectures()}")
    return ARCHITECTURES[canon]["build"](num_classes, fc_dropout, pretrained)


def set_training_phase(
    model: nn.Module,
    architecture: str,
    phase: str,
    *,
    finetune_scope: str = "last_block",
) -> None:
    """head: classifier only; finetune: last_block (default) or full network."""
    canon = architecture.lower()
    if canon not in ARCHITECTURES:
        raise ValueError(architecture)
    if phase == "head":
        ARCHITECTURES[canon]["freeze_head"](model)
    elif phase == "finetune":
        if finetune_scope == "full":
            for param in model.parameters():
                param.requires_grad = True
        elif finetune_scope == "last_block":
            ARCHITECTURES[canon]["unfreeze_finetune"](model)
        else:
            raise ValueError(f"Unknown finetune_scope {finetune_scope!r}")
    else:
        raise ValueError(phase)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
