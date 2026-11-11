#!/usr/bin/env python3
"""Export Table 3 — test-set classification metrics (Frontiers-style, reformatted)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


def load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_weighted(data: Dict[str, Any]) -> Tuple[float, float, float]:
    from sklearn.metrics import precision_recall_fscore_support

    m = data["metrics"]
    nd, nc = int(m["support_dog"]), int(m["support_cat"])
    y_true = [0] * nd + [1] * nc
    cm = data["confusion_matrix"]
    y_pred = [0] * int(cm[0][0]) + [1] * int(cm[0][1]) + [0] * int(cm[1][0]) + [1] * int(cm[1][1])
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return float(p), float(r), float(f)


def build_per_class_rows(m: Dict[str, Any]) -> List[Tuple[str, str, str, str, str]]:
    """Feline first (class 1), then canine (class 0) — manuscript order."""
    return [
        (
            "Feline",
            f"{m['precision_cat']:.2f}",
            f"{m['recall_cat']:.2f}",
            f"{m['f1_cat']:.2f}",
            str(int(m["support_cat"])),
        ),
        (
            "Canine",
            f"{m['precision_dog']:.2f}",
            f"{m['recall_dog']:.2f}",
            f"{m['f1_dog']:.2f}",
            str(int(m["support_dog"])),
        ),
    ]


def build_summary_rows(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    m = data["metrics"]
    nd, nc = int(m["support_dog"]), int(m["support_cat"])
    n = nd + nc
    cm = data["confusion_matrix"]
    correct = int(cm[0][0]) + int(cm[1][1])
    wp, wr, wf = compute_weighted(data)

    return [
        ("Total test radiographs, n", str(n)),
        ("Correctly classified, n (%)", f"{correct} ({100 * correct / n:.1f}%)"),
        ("Accuracy (image-level)", f"{m['accuracy']:.3f} ({100 * m['accuracy']:.1f}%)"),
        ("Balanced accuracy", f"{m['balanced_accuracy']:.3f}"),
        (
            "Macro average — precision / recall / F1",
            f"{m['precision_macro']:.3f} / {m['recall_macro']:.3f} / {m['f1_macro']:.3f}",
        ),
        (
            "Weighted average — precision / recall / F1",
            f"{wp:.3f} / {wr:.3f} / {wf:.3f}",
        ),
        ("ROC-AUC (one-vs-rest, cat vs. dog)", f"{m['roc_auc_cat_vs_dog']:.3f}"),
    ]


def shade(cell, hex_fill: str) -> None:
    sh = OxmlElement("w:shd")
    sh.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(sh)


def write_docx(out: Path, per_class: List[Tuple[str, str, str, str, str]], summary: List[Tuple[str, str]]) -> None:
    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r0 = cap.add_run("Table 3 ")
    r0.bold = True
    r0.font.size = Pt(10)
    r1 = cap.add_run(
        "| Image-level classification performance on the held-out test set "
        "(ResNet-18, patient-level split; argmax class from softmax outputs)."
    )
    r1.font.size = Pt(10)

    # --- Per-class table ---
    doc.add_paragraph()
    sub = doc.add_paragraph("Per-class metrics")
    sub.runs[0].bold = True
    sub.runs[0].font.size = Pt(10)

    t1 = doc.add_table(rows=1 + len(per_class), cols=5)
    t1.style = "Table Grid"
    headers = ["Species class", "Precision", "Recall", "F1-score", "Support (n)"]
    for i, h in enumerate(headers):
        t1.rows[0].cells[i].text = h
        shade(t1.rows[0].cells[i], "E7E6E6")
        for p in t1.rows[0].cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)

    for ri, row in enumerate(per_class, start=1):
        for ci, val in enumerate(row):
            t1.rows[ri].cells[ci].text = val
            for p in t1.rows[ri].cells[ci].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
                    if ci == 0:
                        r.bold = True
            if ci > 0:
                for p in t1.rows[ri].cells[ci].paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()
    sub2 = doc.add_paragraph("Aggregate test-set metrics")
    sub2.runs[0].bold = True
    sub2.runs[0].font.size = Pt(10)

    t2 = doc.add_table(rows=len(summary), cols=2)
    t2.style = "Table Grid"
    for ri, (label, val) in enumerate(summary):
        t2.rows[ri].cells[0].text = label
        t2.rows[ri].cells[1].text = val
        shade(t2.rows[ri].cells[0], "F5F5F5")
        for p in t2.rows[ri].cells[0].paragraphs:
            for r in p.runs:
                r.font.size = Pt(9)
                r.bold = True
        for p in t2.rows[ri].cells[1].paragraphs:
            for r in p.runs:
                r.font.size = Pt(9)

    doc.add_paragraph()
    notes = [
        "Metrics computed at the radiograph (image) level on the held-out test partition (29 patients; 89 DICOM instances).",
        "Class labels: canine = 0, feline = 1. Predictions used the argmax of two-class softmax outputs (no threshold tuning on test).",
        "Model checkpoint selected by highest validation accuracy during training.",
        "Confusion matrix (true × predicted): canine 43 correct / 12 misclassified as feline; feline 30 correct / 4 misclassified as canine.",
    ]
    for note in notes:
        p = doc.add_paragraph()
        t = p.add_run(note)
        t.font.size = Pt(8)
        t.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.save(out)


def write_markdown(out: Path, per_class: List[Tuple[str, str, str, str, str]], summary: List[Tuple[str, str]]) -> None:
    lines = [
        "**Table 3** | Image-level classification performance on the held-out test set "
        "(ResNet-18; patient-level split).",
        "",
        "### Per-class metrics",
        "",
        "| Species class | Precision | Recall | F1-score | Support (n) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in per_class:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "### Aggregate test-set metrics", "", "| Metric | Value |", "| --- | --- |"])
    for label, val in summary:
        lines.append(f"| {label} | {val} |")
    lines.extend(
        [
            "",
            "Notes: image-level evaluation on 89 test radiographs; argmax prediction; "
            "checkpoint by best validation accuracy.",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="/home/miorga/Doctorat/test1/metrics_1.json")
    parser.add_argument("--output_dir", default="/home/miorga/Doctorat/test1")
    args = parser.parse_args()
    data = load_metrics(Path(args.metrics))
    per_class = build_per_class_rows(data["metrics"])
    summary = build_summary_rows(data)
    out_dir = Path(args.output_dir)
    docx = out_dir / "Table_3_Classification_performance_Frontiers.docx"
    md = out_dir / "Table_3_Classification_performance_Frontiers.md"
    write_docx(docx, per_class, summary)
    write_markdown(md, per_class, summary)
    print(f"Wrote {docx}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
