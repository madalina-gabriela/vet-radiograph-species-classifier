#!/usr/bin/env python3
"""Export dataset demographics table to Word (.docx) and Markdown from dataset_statistics.json."""

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


def load_data(json_path: Path) -> Dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def pct(n: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{100 * n / denom:.1f}%"


def fmt_median_range(demo: Dict[str, Any]) -> str:
    if demo.get("n", 0) == 0:
        return "Not recorded"
    return f"{demo['median']:.1f} ({demo['min']:.1f}–{demo['max']:.1f})"


def sex_cell(sex_map: Dict[str, int], n_patients: int) -> str:
    order = ["Male", "Female", "Male (neutered)", "Female (spayed)", "Unknown", "Other"]
    parts = []
    for k in order:
        if sex_map.get(k):
            parts.append(f"{k}: {sex_map[k]} ({pct(sex_map[k], n_patients)})")
    for k, v in sex_map.items():
        if k not in order and v:
            parts.append(f"{k}: {v} ({pct(v, n_patients)})")
    return "; ".join(parts) if parts else "Not recorded"


def breed_cell(breed_list: List[List[Any]], n_images: int) -> str:
    from collections import Counter

    counts = Counter()
    for b, n in breed_list:
        counts[b] += int(n)
    mixed = counts.get("Mixed / common (no breed specified)", 0)
    placeholder = counts.get("Not recorded (placeholder tag)", 0)
    named = max(0, n_images - mixed - placeholder)
    lines = [
        f"Mixed/common: {mixed} ({pct(mixed, n_images)})",
        f"Missing/placeholder tag: {placeholder} ({pct(placeholder, n_images)})",
        f"Named label: {named} ({pct(named, n_images)})",
    ]
    top = [(b, n) for b, n in counts.most_common() if b not in ("Mixed / common (no breed specified)", "Not recorded (placeholder tag)")][:3]
    if top:
        lines.append("Top named: " + "; ".join(f"{b} (n={n})" for b, n in top))
    return "\n".join(lines)


def body_size_cell(cats: Dict[str, int], n: int) -> str:
    if not cats or n == 0:
        return "Not recorded"
    return "; ".join(f"{k}: {v} ({pct(v, n)})" for k, v in sorted(cats.items(), key=lambda x: -x[1]))


def row_from_manuscript(data: Dict[str, Any], prefix: str) -> Tuple[str, str, str]:
    for row in data["manuscript_table"]:
        if row["Characteristic"].startswith(prefix):
            return row["Overall"], row["Canine"], row["Feline"]
    return "—", "—", "—"


def build_rows(data: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    dog_p = data["counts"]["dog"]["patients"]
    cat_p = data["counts"]["cat"]["patients"]
    dog_i = data["counts"]["dog"]["images"]
    cat_i = data["counts"]["cat"]["images"]
    all_p = dog_p + cat_p
    all_i = dog_i + cat_i
    ipp = data["images_per_patient"]
    dd = data["demographics_dog"]
    dc = data["demographics_cat"]

    multi_dog = 99
    multi_cat = 78
    multi_all = multi_dog + multi_cat

    age_all = (
        f"Dogs {fmt_median_range(dd['age_years'])} (n={dd['age_years']['n']}); "
        f"Cats {fmt_median_range(dc['age_years'])} (n={dc['age_years']['n']})"
    )

    reg_o, reg_d, reg_c = row_from_manuscript(data, "Anatomical region")
    proj_o, proj_d, proj_c = row_from_manuscript(data, "Projection")

    equip = "Intech-Thales Pixium 4343REc digital radiography system"
    quality = (
        "2880×2880 matrix; 16-bit storage; LossyImageCompression absent (100%); "
        "clinical inclusion without severe motion/blur/artifacts"
    )

    return [
        ("Individual animals, n", str(all_p), str(dog_p), str(cat_p)),
        ("Radiographs (DICOM instances), n", str(all_i), str(dog_i), str(cat_i)),
        (
            "Images per animal, median (range)",
            f"{ipp['median']} ({ipp['min']}–{ipp['max']})",
            f"{ipp['median']} ({ipp['min']}–{ipp['max']})",
            "3 (1–9)",
        ),
        (
            "Animals with >1 radiograph, n (%)",
            f"{multi_all} ({pct(multi_all, all_p)})",
            f"{multi_dog} ({pct(multi_dog, dog_p)})",
            f"{multi_cat} ({pct(multi_cat, cat_p)})",
        ),
        (
            "Age at imaging (years), median (range)a",
            age_all,
            f"{fmt_median_range(dd['age_years'])}; n={dd['age_years']['n']} patients",
            f"{fmt_median_range(dc['age_years'])}; n={dc['age_years']['n']} patients",
        ),
        (
            "Sex (patients)b",
            f"Dogs — {sex_cell(dd['sex_among_patients'], dog_p)}; Cats — {sex_cell(dc['sex_among_patients'], cat_p)}",
            sex_cell(dd["sex_among_patients"], dog_p),
            sex_cell(dc["sex_among_patients"], cat_p),
        ),
        (
            "Breed or variety (DICOM PatientBirthName)c",
            breed_cell(data["breed_counts"]["dog"] + data["breed_counts"]["cat"], all_i),
            breed_cell(data["breed_counts"]["dog"], dog_i),
            breed_cell(data["breed_counts"]["cat"], cat_i),
        ),
        (
            "Body weight (kg), median (range)a",
            "7.0 (1.0–60.0); n=565 images with weight tag",
            f"{fmt_median_range(dd['weight_kg'])}; n={dd['weight_kg']['n']} patients",
            f"{fmt_median_range(dc['weight_kg'])}; n={dc['weight_kg']['n']} patients",
        ),
        (
            "Body size proxy (from weight)d",
            body_size_cell({**dd["body_size_categories"], **dc["body_size_categories"]}, dd["weight_kg"]["n"] + dc["weight_kg"]["n"]),
            body_size_cell(dd["body_size_categories"], dd["weight_kg"]["n"]),
            body_size_cell(dc["body_size_categories"], dc["weight_kg"]["n"]),
        ),
        ("Anatomical region (image-level)e", reg_o, reg_d, reg_c),
        ("Projection (image-level)e", proj_o, proj_d, proj_c),
        ("Modality", "Digital radiography (DX)", "DX", "DX"),
        ("Imaging equipment", equip, equip, equip),
        ("Image quality and format", quality, quality, quality),
        (
            "Multiple views from the same animal",
            "Included (all instances per patient folder)",
            "Included",
            "Included",
        ),
        (
            "Model development splitf",
            "70/15/15 stratified by patient (434/69/89 images; 130/28/29 patients)",
            "Patient-level (no leakage)",
            "Patient-level (no leakage)",
        ),
    ]


def set_cell_shading(cell, fill_hex: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill_hex)
    cell._tc.get_or_add_tcPr().append(shading)


def write_docx(out_path: Path, rows: List[Tuple[str, str, str, str]]) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(10)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r0 = cap.add_run("Table 1 ")
    r0.bold = True
    r0.font.name = "Arial"
    r0.font.size = Pt(10)
    r1 = cap.add_run("| Dataset characteristics of digital radiographs from dogs and cats included in the deep learning study.")
    r1.font.name = "Arial"
    r1.font.size = Pt(10)

    headers = [
        "Characteristic",
        "Overall\n(187 animals; 592 images)",
        "Canine\n(103 animals; 366 images)",
        "Feline\n(84 animals; 226 images)",
    ]
    table = doc.add_table(rows=1 + len(rows), cols=4)
    table.style = "Table Grid"

    for i, text in enumerate(headers):
        table.rows[0].cells[i].text = text
        set_cell_shading(table.rows[0].cells[i], "E7E6E6")
        for p in table.rows[0].cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.name = "Arial"
                r.font.size = Pt(9)

    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            table.rows[ri].cells[ci].text = val
            for p in table.rows[ri].cells[ci].paragraphs:
                for r in p.runs:
                    r.font.name = "Arial"
                    r.font.size = Pt(9)
            if ci == 0:
                for p in table.rows[ri].cells[ci].paragraphs:
                    for r in p.runs:
                        r.bold = True

    doc.add_paragraph()
    footnotes = [
        "Age and body weight from DICOM PatientBirthDate, StudyDate, and PatientWeight; patient-level medians exclude missing or implausible values.",
        "Sex from DICOM PatientSex (MC = neutered male).",
        "Breed from PatientBirthName (hospital RIS); mixed/common labels grouped; pedigree not independently verified.",
        "Body size from weight using species-specific thresholds (dogs: <10, 10–25, ≥25 kg; cats: <4, 4–6, ≥6 kg).",
        "Region and projection from StudyDescription/BodyPartExamined (image level).",
        "All images from one animal retained; train/validation/test split stratified by patient folder.",
    ]
    for i, fn in enumerate(footnotes):
        p = doc.add_paragraph()
        s = p.add_run(chr(ord("a") + i))
        s.font.superscript = True
        s.font.name = "Arial"
        s.font.size = Pt(8)
        t = p.add_run(fn)
        t.font.name = "Arial"
        t.font.size = Pt(8)
        t.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    doc.save(out_path)


def write_markdown(out_path: Path, rows: List[Tuple[str, str, str, str]]) -> None:
    lines = [
        "**Table 1** | Dataset characteristics of digital radiographs from dogs and cats included in the deep learning study.",
        "",
        "| Characteristic | Overall (187 animals; 592 images) | Canine (103 animals; 366 images) | Feline (84 animals; 226 images) |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        cells = [c.replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Footnotes: *a* age/weight; *b* sex; *c* breed (RIS); *d* size categories; *e* image-level anatomy; *f* patient-level split."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default="/home/miorga/Doctorat/test1/dataset_characteristics/dataset_characteristics.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/miorga/Doctorat/test1/dataset_characteristics",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    data = load_data(Path(args.input_json))
    rows = build_rows(data)
    docx_path = out_dir / "Table_1_Dataset_characteristics.docx"
    md_path = out_dir / "Table_1_Dataset_characteristics.md"
    write_docx(docx_path, rows)
    write_markdown(md_path, rows)
    print(f"Wrote {docx_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
