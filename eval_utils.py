"""Evaluation helpers: patient-level aggregation and metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def aggregate_predictions_by_patient(
    patient_keys: List[str],
    y_true: np.ndarray,
    y_prob_cat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean P(cat) per patient, then threshold at 0.5."""
    prob_lists: Dict[str, List[float]] = defaultdict(list)
    label_by_patient: Dict[str, int] = {}
    for pk, yt, pc in zip(patient_keys, y_true.tolist(), y_prob_cat.tolist()):
        prob_lists[pk].append(float(pc))
        label_by_patient[pk] = int(yt)

    ordered_keys = sorted(prob_lists.keys())
    y_true_p: List[int] = []
    y_prob_cat_p: List[float] = []
    y_pred_p: List[int] = []
    for pk in ordered_keys:
        mean_cat = float(np.mean(prob_lists[pk]))
        yt = label_by_patient[pk]
        y_true_p.append(yt)
        y_prob_cat_p.append(mean_cat)
        y_pred_p.append(1 if mean_cat >= 0.5 else 0)

    return (
        np.asarray(y_true_p, dtype=int),
        np.asarray(y_pred_p, dtype=int),
        np.asarray(y_prob_cat_p, dtype=np.float32),
    )


def patient_prediction_rows(
    patient_keys: List[str],
    y_true: np.ndarray,
    y_prob_cat: np.ndarray,
    class_name: Dict[int, str],
) -> List[Dict[str, Any]]:
    prob_lists: Dict[str, List[float]] = defaultdict(list)
    label_by_patient: Dict[str, int] = {}
    for pk, yt, pc in zip(patient_keys, y_true.tolist(), y_prob_cat.tolist()):
        prob_lists[pk].append(float(pc))
        label_by_patient[pk] = int(yt)

    rows: List[Dict[str, Any]] = []
    for pk in sorted(prob_lists.keys()):
        mean_cat = float(np.mean(prob_lists[pk]))
        mean_dog = 1.0 - mean_cat
        yt = label_by_patient[pk]
        pred = 1 if mean_cat >= 0.5 else 0
        rows.append(
            {
                "patient_key": pk,
                "n_images": len(prob_lists[pk]),
                "true_label_id": yt,
                "true_label_name": class_name[yt],
                "pred_label_id": pred,
                "pred_label_name": class_name[pred],
                "prob_dog": mean_dog,
                "prob_cat": mean_cat,
                "correct": bool(yt == pred),
            }
        )
    return rows
