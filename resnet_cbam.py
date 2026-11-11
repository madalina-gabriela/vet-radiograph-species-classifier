"""ResNet-18 with CBAM after layer4 (512 channels)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from cbam import CBAM
from model_utils import _adapt_grayscale_conv, build_classifier_head


class ResNet18CBAM(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        fc_dropout: float = 0.0,
        weights: Optional[ResNet18_Weights] = ResNet18_Weights.DEFAULT,
    ) -> None:
        super().__init__()
        base = models.resnet18(weights=weights)
        self.conv1 = _adapt_grayscale_conv(base.conv1)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.cbam = CBAM(512)
        self.avgpool = base.avgpool
        self.fc = build_classifier_head(base.fc.in_features, num_classes, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
