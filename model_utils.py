"""CNN builders and checkpoint loading for cat/dog DICOM classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights, ResNet18_Weights


def build_classifier_head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    return nn.Linear(in_features, num_classes)


def build_resnet18_catdog(num_classes: int = 2, fc_dropout: float = 0.0) -> nn.Module:
    model = models.resnet18(weights=None)
    if fc_dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=False,
    )
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {"model_state_dict": checkpoint}


def infer_fc_dropout(state_dict: Dict[str, torch.Tensor]) -> float:
    if any(k.startswith("fc.1.weight") or k.startswith("classifier.1.weight") for k in state_dict):
        return 0.3
    return 0.0


def _adapt_grayscale_conv(pretrained_conv: nn.Conv2d) -> nn.Conv2d:
    conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=pretrained_conv.out_channels,
        kernel_size=pretrained_conv.kernel_size,
        stride=pretrained_conv.stride,
        padding=pretrained_conv.padding,
        bias=pretrained_conv.bias is not None,
    )
    with torch.no_grad():
        conv1.weight.copy_(pretrained_conv.weight.mean(dim=1, keepdim=True))
        if conv1.bias is not None and pretrained_conv.bias is not None:
            conv1.bias.copy_(pretrained_conv.bias)
    return conv1


def build_densenet121_catdog(
    num_classes: int = 2,
    fc_dropout: float = 0.0,
    weights: Optional[DenseNet121_Weights] = DenseNet121_Weights.DEFAULT,
) -> nn.Module:
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    if fc_dropout > 0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.classifier = nn.Linear(in_features, num_classes)

    model.features.conv0 = _adapt_grayscale_conv(model.features.conv0)
    return model
