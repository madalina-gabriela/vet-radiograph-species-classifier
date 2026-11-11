#!/usr/bin/env python3
"""
Post-hoc statistics from saved test predictions (no retraining).

- Cluster bootstrap (resample patients) for 95% CIs on image- and patient-level metrics.
- Paired McNemar (exact binomial) and paired bootstrap on patient-level correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import binom
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from experiment_registry import EXPERIMENTS, list_experiments


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_cat: np.ndarray,
) -> Dict[str, Any]:
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(np.mean(f1)),
    }
    try:
        metrics["roc_auc_cat_vs_dog"] = float(roc_auc_score(y_true, y_prob_cat))
    except Exception:
        metrics["roc_auc_cat_vs_dog"] = None
    return metrics


def load_image_predictions(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_to_patient_level(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict

    probs: Dict[str, List[float]] = defaultdict(list)
    labels: Dict[str, int] = {}
    for r in rows:
        pk = r["patient_key"]
        probs[pk].append(float(r["prob_cat"]))
        labels[pk] = int(r["true_label_id"])
    out: List[Dict[str, Any]] = []
    for pk in sorted(probs.keys()):
        mean_cat = float(np.mean(probs[pk]))
        yt = labels[pk]
        pred = 1 if mean_cat >= 0.5 else 0
        out.append(
            {
                "patient_key": pk,
                "n_images": len(probs[pk]),
                "true_label_id": yt,
                "pred_label_id": pred,
                "prob_cat": mean_cat,
                "correct": bool(yt == pred),
            }
        )
    return out


def _arrays_from_images(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([int(r["true_label_id"]) for r in rows], dtype=int)
    y_pred = np.array([int(r["pred_label_id"]) for r in rows], dtype=int)
    y_prob = np.array([float(r["prob_cat"]) for r in rows], dtype=np.float32)
    return y_true, y_pred, y_prob


def _arrays_from_patients(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([int(r["true_label_id"]) for r in rows], dtype=int)
    y_pred = np.array([int(r["pred_label_id"]) for r in rows], dtype=int)
    y_prob = np.array([float(r["prob_cat"]) for r in rows], dtype=np.float32)
    return y_true, y_pred, y_prob


def build_cluster_bootstrap_plan(
    image_rows: List[Dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
    """Shared patient resampling for all experiments (same test patients/images)."""
    rng = np.random.default_rng(seed)
    patients = sorted({r["patient_key"] for r in image_rows})
    by_pk: Dict[str, List[int]] = {pk: [] for pk in patients}
    for i, r in enumerate(image_rows):
        by_pk[r["patient_key"]].append(i)
    img_slices = [np.asarray(by_pk[pk], dtype=int) for pk in patients]
    samples = rng.integers(0, len(patients), size=(n_boot, len(patients)))
    img_idx_boot = [np.concatenate([img_slices[j] for j in samples[b]]) for b in range(n_boot)]
    return patients, img_idx_boot, samples


def cluster_bootstrap_metrics(
    patient_rows: List[Dict[str, Any]],
    image_rows: List[Dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
    bootstrap_plan: Tuple[List[str], List[np.ndarray], np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Resample patients; image metrics use all images from sampled patients."""
    if bootstrap_plan is None:
        patients, img_idx_boot, samples = build_cluster_bootstrap_plan(
            image_rows, n_boot=n_boot, seed=seed
        )
    else:
        patients, img_idx_boot, samples = bootstrap_plan

    p_by_key = {r["patient_key"]: r for r in patient_rows}
    assert set(p_by_key.keys()) == set(patients)

    yt_i_all, yp_i_all, yprob_i_all = _arrays_from_images(image_rows)

    yt_p = np.array([p_by_key[pk]["true_label_id"] for pk in patients], dtype=int)
    yp_p = np.array([p_by_key[pk]["pred_label_id"] for pk in patients], dtype=int)
    yprob_p = np.array([p_by_key[pk]["prob_cat"] for pk in patients], dtype=np.float32)

    alpha = (1.0 - ci) / 2.0
    lo_q, hi_q = 100 * alpha, 100 * (1.0 - alpha)

    n_pat = len(patients)
    n_boot = int(samples.shape[0])

    img_draws: Dict[str, List[float]] = {k: [] for k in ("accuracy", "balanced_accuracy", "f1_macro")}
    pat_draws: Dict[str, List[float]] = {k: [] for k in ("accuracy", "balanced_accuracy", "f1_macro")}

    pat_correct = (yt_p == yp_p).astype(np.float64)
    for b in range(n_boot):
        s = samples[b]
        yt_s, yp_s = yt_p[s], yp_p[s]
        pat_draws["accuracy"].append(float(pat_correct[s].mean()))
        pat_draws["f1_macro"].append(float(f1_score(yt_s, yp_s, average="macro", zero_division=0)))
        pat_draws["balanced_accuracy"].append(float(balanced_accuracy_score(yt_s, yp_s)))

        idx_img = img_idx_boot[b]
        yt_i, yp_i = yt_i_all[idx_img], yp_i_all[idx_img]
        img_draws["accuracy"].append(float((yt_i == yp_i).mean()))
        img_draws["f1_macro"].append(float(f1_score(yt_i, yp_i, average="macro", zero_division=0)))
        img_draws["balanced_accuracy"].append(float(balanced_accuracy_score(yt_i, yp_i)))

    def _summarize(draws: List[float], point: float) -> Dict[str, float]:
        if not draws:
            return {"point": point, "ci_low": None, "ci_high": None}
        arr = np.asarray(draws, dtype=float)
        return {
            "point": point,
            "ci_low": float(np.percentile(arr, lo_q)),
            "ci_high": float(np.percentile(arr, hi_q)),
        }

    yt_i, yp_i, yprob_i = _arrays_from_images(image_rows)
    yt_p, yp_p, yprob_p = _arrays_from_patients(patient_rows)
    point_i = compute_metrics(yt_i, yp_i, yprob_i)
    point_p = compute_metrics(yt_p, yp_p, yprob_p)

    key_to_metric = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }

    def _level_summary(draws_map: Dict[str, List[float]], point: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, mk in key_to_metric.items():
            val = point.get(mk)
            pt = float(val) if val is not None else None
            out[k] = _summarize(draws_map[k], pt if pt is not None else float("nan"))
            if pt is not None:
                out[k]["point"] = pt
        auc = point.get("roc_auc_cat_vs_dog")
        out["roc_auc"] = {
            "point": float(auc) if auc is not None else None,
            "ci_low": None,
            "ci_high": None,
            "note": "AUC point estimate only; bootstrap CI omitted for speed",
        }
        return out

    return {
        "n_bootstrap": n_boot,
        "ci_level": ci,
        "n_patients": len(patients),
        "n_images": len(image_rows),
        "image_level": _level_summary(img_draws, point_i),
        "patient_level": _level_summary(pat_draws, point_p),
    }


def mcnemar_exact(n10: int, n01: int) -> Dict[str, Any]:
    """n10: A correct B wrong; n01: A wrong B correct."""
    n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "n_discordant": 0, "p_value_two_sided": 1.0}
    k = min(n10, n01)
    p = float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))
    return {"n10": int(n10), "n01": int(n01), "n_discordant": int(n), "p_value_two_sided": p}


def paired_image_cluster_bootstrap(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    bootstrap_plan: Tuple[List[str], List[np.ndarray], np.ndarray],
) -> Dict[str, Any]:
    if len(rows_a) != len(rows_b):
        raise ValueError("Row count mismatch between models")
    for ra, rb in zip(rows_a, rows_b):
        if ra["image_path"] != rb["image_path"]:
            raise ValueError("Image paths not aligned between models")
    yt = np.array([int(r["true_label_id"]) for r in rows_a], dtype=int)
    ypa = np.array([int(r["pred_label_id"]) for r in rows_a], dtype=int)
    ypb = np.array([int(r["pred_label_id"]) for r in rows_b], dtype=int)

    _patients, img_idx_boot, _samples = bootstrap_plan

    acc_diffs: List[float] = []
    f1_diffs: List[float] = []
    for idx in img_idx_boot:
        if idx.size == 0:
            continue
        yts, ya, yb = yt[idx], ypa[idx], ypb[idx]
        acc_diffs.append(float(np.mean(ya == yts) - np.mean(yb == yts)))
        f1_diffs.append(
            float(f1_score(yts, ya, average="macro", zero_division=0) - f1_score(yts, yb, average="macro", zero_division=0))
        )
    alpha = 0.025
    lo_q, hi_q = 100 * alpha, 100 * (1 - alpha)

    def _summ(diffs: List[float], point: float) -> Dict[str, float]:
        arr = np.asarray(diffs, dtype=float)
        p_two = min(1.0, float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))))
        if arr.size and np.all(arr == 0):
            p_two = 1.0
        return {
            "point_diff_a_minus_b": point,
            "ci_low": float(np.percentile(arr, lo_q)),
            "ci_high": float(np.percentile(arr, hi_q)),
            "p_value_two_sided_bootstrap": p_two,
        }

    acc_pt = float(np.mean(ypa == yt) - np.mean(ypb == yt))
    f1_pt = float(f1_score(yt, ypa, average="macro", zero_division=0) - f1_score(yt, ypb, average="macro", zero_division=0))
    return {
        "accuracy": _summ(acc_diffs, acc_pt),
        "f1_macro": _summ(f1_diffs, f1_pt),
    }


def paired_patient_tests(
    pat_a: List[Dict[str, Any]],
    pat_b: List[Dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
    bootstrap_samples: np.ndarray | None = None,
) -> Dict[str, Any]:
    keys_a = {r["patient_key"]: r for r in pat_a}
    keys_b = {r["patient_key"]: r for r in pat_b}
    common = sorted(set(keys_a.keys()) & set(keys_b.keys()))
    if len(common) != len(keys_a) or len(common) != len(keys_b):
        raise ValueError("Patient keys differ between models — check test split alignment.")

    correct_a = np.array([bool(keys_a[pk]["correct"]) for pk in common], dtype=bool)
    correct_b = np.array([bool(keys_b[pk]["correct"]) for pk in common], dtype=bool)
    yt = np.array([int(keys_a[pk]["true_label_id"]) for pk in common], dtype=int)
    yp_a = np.array([int(keys_a[pk]["pred_label_id"]) for pk in common], dtype=int)
    yp_b = np.array([int(keys_b[pk]["pred_label_id"]) for pk in common], dtype=int)
    prob_a = np.array([float(keys_a[pk]["prob_cat"]) for pk in common], dtype=np.float32)
    prob_b = np.array([float(keys_b[pk]["prob_cat"]) for pk in common], dtype=np.float32)

    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    mcn = mcnemar_exact(n10, n01)

    alpha = (1.0 - ci) / 2.0
    lo_q, hi_q = 100 * alpha, 100 * (1.0 - alpha)

    acc_diffs: List[float] = []
    f1_diffs: List[float] = []
    bal_diffs: List[float] = []
    n_pat = len(common)
    if bootstrap_samples is not None:
        if bootstrap_samples.shape[1] != n_pat:
            raise ValueError("bootstrap_samples width must match n_patients")
        sample_rows = bootstrap_samples
        n_boot = int(sample_rows.shape[0])
    else:
        rng = np.random.default_rng(seed)
        sample_rows = rng.integers(0, n_pat, size=(n_boot, n_pat))

    for b in range(n_boot):
        idx = sample_rows[b]
        ca = correct_a[idx]
        cb = correct_b[idx]
        acc_diffs.append(float(np.mean(ca) - np.mean(cb)))
        yt_s, ypa_s, ypb_s = yt[idx], yp_a[idx], yp_b[idx]
        f1_diffs.append(
            float(
                f1_score(yt_s, ypa_s, average="macro", zero_division=0)
                - f1_score(yt_s, ypb_s, average="macro", zero_division=0)
            )
        )
        bal_diffs.append(
            float(balanced_accuracy_score(yt_s, ypa_s) - balanced_accuracy_score(yt_s, ypb_s))
        )

    def _boot_summary(diffs: List[float], point: float) -> Dict[str, float]:
        arr = np.asarray(diffs, dtype=float)
        p_two = float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0)))
        p_two = min(1.0, p_two)
        if arr.size and np.all(arr == 0):
            p_two = 1.0
        return {
            "point_diff_a_minus_b": point,
            "ci_low": float(np.percentile(arr, lo_q)),
            "ci_high": float(np.percentile(arr, hi_q)),
            "p_value_two_sided_bootstrap": p_two,
        }

    acc_point = float(np.mean(correct_a) - np.mean(correct_b))
    f1_point = float(
        f1_score(yt, yp_a, average="macro", zero_division=0)
        - f1_score(yt, yp_b, average="macro", zero_division=0)
    )
    bal_point = float(balanced_accuracy_score(yt, yp_a) - balanced_accuracy_score(yt, yp_b))

    return {
        "n_patients": n_pat,
        "mcnemar_correctness": mcn,
        "bootstrap_patient_level": {
            "accuracy": _boot_summary(acc_diffs, acc_point),
            "balanced_accuracy": _boot_summary(bal_diffs, bal_point),
            "f1_macro": _boot_summary(f1_diffs, f1_point),
        },
    }


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def mcnemar_images(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_path_a = {r["image_path"]: r for r in rows_a}
    by_path_b = {r["image_path"]: r for r in rows_b}
    common = sorted(set(by_path_a.keys()) & set(by_path_b.keys()))
    n10 = n01 = 0
    for p in common:
        ca = _as_bool(by_path_a[p]["correct"])
        cb = _as_bool(by_path_b[p]["correct"])
        if ca and not cb:
            n10 += 1
        elif not ca and cb:
            n01 += 1
    out = mcnemar_exact(n10, n01)
    out["n_images"] = len(common)
    return out


def load_experiment(root: Path, code: str) -> Dict[str, Any]:
    csv_path = root / code / "best" / "predictions_detailed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    img = load_image_predictions(csv_path)
    pat = rows_to_patient_level(img)
    return {"code": code, "image_rows": img, "patient_rows": pat}


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap CIs and paired tests from saved predictions")
    p.add_argument("--comparison_dir", type=str, default="/home/miorga/Doctorat/test_comparison")
    p.add_argument("--n_bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline", type=str, default="M0")
    p.add_argument("--experiments", type=str, default="", help="Comma-separated; default all with predictions")
    args = p.parse_args()
    root = Path(args.comparison_dir)
    if args.experiments.strip():
        codes = [c.strip().upper() for c in args.experiments.split(",")]
    else:
        codes = [c for c in list_experiments() if (root / c / "best" / "predictions_detailed.csv").exists()]

    loaded: Dict[str, Dict[str, Any]] = {}
    for code in codes:
        loaded[code] = load_experiment(root, code)

    ref_img = loaded[codes[0]]["image_rows"]
    boot_plan = build_cluster_bootstrap_plan(ref_img, n_boot=args.n_bootstrap, seed=args.seed)
    _patients_ref, _img_idx, shared_patient_samples = boot_plan

    per_exp: Dict[str, Any] = {}
    for code, data in loaded.items():
        pat = data["patient_rows"]
        img = data["image_rows"]
        boot = cluster_bootstrap_metrics(
            pat, img, n_boot=args.n_bootstrap, seed=args.seed, bootstrap_plan=boot_plan
        )
        per_exp[code] = {
            "title": EXPERIMENTS.get(code, {}).get("title", code),
            "bootstrap_cluster_by_patient": boot,
        }

    baseline = args.baseline.upper()
    if baseline not in loaded:
        raise SystemExit(f"Baseline {baseline} not in loaded experiments")

    pairwise: Dict[str, Any] = {}
    pat_b = loaded[baseline]["patient_rows"]
    img_b = loaded[baseline]["image_rows"]
    for code in codes:
        if code == baseline:
            continue
        pairwise[f"{code}_vs_{baseline}"] = {
            "patient_level": paired_patient_tests(
                loaded[code]["patient_rows"],
                pat_b,
                n_boot=args.n_bootstrap,
                seed=args.seed + hash(code) % 10000,
                bootstrap_samples=shared_patient_samples,
            ),
            "image_level_mcnemar": mcnemar_images(loaded[code]["image_rows"], img_b),
            "image_level_bootstrap_cluster": paired_image_cluster_bootstrap(
                loaded[code]["image_rows"],
                img_b,
                boot_plan,
            ),
        }

    payload = {
        "method": {
            "bootstrap": f"cluster (patient) resampling, n={args.n_bootstrap}, seed={args.seed}, 95% percentile CI",
            "paired_tests": "McNemar exact (binomial) on discordant pairs; paired bootstrap differences use the same patient resamples as single-model CIs",
            "data": "test set predictions from M*/best/predictions_detailed.csv",
            "patient_prediction": "mean prob_cat per patient, threshold 0.5",
        },
        "experiments": per_exp,
        "pairwise_vs_baseline": pairwise,
        "baseline": baseline,
    }

    out_json = root / "comparison_statistics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _write_markdown_summary(root / "comparison_statistics.md", payload, codes)
    print(f"Wrote {out_json}")
    print(f"Wrote {root / 'comparison_statistics.md'}")


def _write_markdown_summary(path: Path, payload: Dict[str, Any], codes: List[str]) -> None:
    lines = [
        "# Statistici post-hoc (test, fără reantrenare)",
        "",
        payload["method"]["bootstrap"],
        "",
        "## Intervale de încredere 95% (bootstrap pe pacienți)",
        "",
        "### Nivel pacient (n≈29)",
        "",
        "| Cod | Acc. point | Acc. CI | Macro-F1 point | Macro-F1 CI | Bal. acc. CI |",
        "|-----|------------|---------|----------------|-------------|--------------|",
    ]
    for code in codes:
        pl = payload["experiments"][code]["bootstrap_cluster_by_patient"]["patient_level"]
        acc, f1, bal = pl["accuracy"], pl["f1_macro"], pl["balanced_accuracy"]
        lines.append(
            f"| {code} | {100*acc['point']:.1f}% | [{100*acc['ci_low']:.1f}, {100*acc['ci_high']:.1f}] | "
            f"{100*f1['point']:.1f}% | [{100*f1['ci_low']:.1f}, {100*f1['ci_high']:.1f}] | "
            f"[{100*bal['ci_low']:.1f}, {100*bal['ci_high']:.1f}] |"
        )
    lines.extend(
        [
            "",
            "### Nivel imagine (n=89, cluster bootstrap)",
            "",
            "| Cod | Acc. CI | Macro-F1 CI |",
            "|-----|---------|-------------|",
        ]
    )
    for code in codes:
        il = payload["experiments"][code]["bootstrap_cluster_by_patient"]["image_level"]
        acc, f1 = il["accuracy"], il["f1_macro"]
        lines.append(
            f"| {code} | [{100*acc['ci_low']:.1f}, {100*acc['ci_high']:.1f}] | "
            f"[{100*f1['ci_low']:.1f}, {100*f1['ci_high']:.1f}] |"
        )

    base = payload["baseline"]
    lines.extend(["", f"## Perechi vs {base} (McNemar + bootstrap diferență)", ""])
    for key, val in sorted(payload["pairwise_vs_baseline"].items()):
        code = key.split("_vs_")[0]
        pl = val["patient_level"]
        m = pl["mcnemar_correctness"]
        bd = pl["bootstrap_patient_level"]["accuracy"]
        im = val["image_level_mcnemar"]
        lines.append(f"### {code} vs {base}")
        lines.append(
            f"- Pacient McNemar: n10={m['n10']}, n01={m['n01']}, p={m['p_value_two_sided']:.4f}"
        )
        lines.append(
            f"- Pacient Δacc ({code}−{base}): {100*bd['point_diff_a_minus_b']:+.1f}% "
            f"CI [{100*bd['ci_low']:+.1f}, {100*bd['ci_high']:+.1f}], bootstrap p={bd['p_value_two_sided_bootstrap']:.4f}"
        )
        bf1 = pl["bootstrap_patient_level"]["f1_macro"]
        lines.append(
            f"- Pacient Δmacro-F1: {100*bf1['point_diff_a_minus_b']:+.1f} pp "
            f"CI [{100*bf1['ci_low']:+.1f}, {100*bf1['ci_high']:+.1f}], bootstrap p={bf1['p_value_two_sided_bootstrap']:.4f}"
        )
        lines.append(
            f"- Imagine McNemar: n10={im['n10']}, n01={im['n01']}, p={im['p_value_two_sided']:.4f} (n={im['n_images']})"
        )
        ib = val.get("image_level_bootstrap_cluster", {})
        if ib:
            bd = ib["accuracy"]
            bf = ib["f1_macro"]
            lines.append(
                f"- Imagine Δacc ({code}−{base}): {100*bd['point_diff_a_minus_b']:+.1f}% "
                f"CI [{100*bd['ci_low']:+.1f}, {100*bd['ci_high']:+.1f}], bootstrap p={bd['p_value_two_sided_bootstrap']:.4f}"
            )
            lines.append(
                f"- Imagine Δmacro-F1: {100*bf['point_diff_a_minus_b']:+.1f} pp "
                f"CI [{100*bf['ci_low']:+.1f}, {100*bf['ci_high']:+.1f}], bootstrap p={bf['p_value_two_sided_bootstrap']:.4f}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
