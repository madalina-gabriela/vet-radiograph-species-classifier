"""Shared training utilities (macro-F1 checkpointing, timing)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float]:
    """Returns (mean loss, accuracy, macro-F1) on the loader.

    Loss is averaged over samples (sum of batch losses weighted by batch size).
    Accuracy and macro-F1 are computed **once** from all predictions in the loader
    (sklearn ``f1_score(..., average=\"macro\")``), not as a mean of per-batch F1.
    """
    model.eval()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.detach().cpu().item()) * y.size(0)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
        n += int(y.size(0))

    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if n else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if n else 0.0
    return total_loss / max(1, n), acc, macro_f1
