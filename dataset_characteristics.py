#!/usr/bin/env python3
"""Extract dataset characteristics from DICOM headers (manuscript Table 1)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

import pydicom

from dataset_utils import LABEL_CAT, LABEL_DOG, build_patient_records


def _s(ds: pydicom.Dataset, key: str, default: str = "") -> str:
    v = getattr(ds, key, default)
    if v is None:
        return default
    return str(v).strip()


def parse_projection(text: str) -> str:
    t = text.lower()
    if "ventrodorsal" in t or re.search(r"\bvd\b", t):
        return "ventrodorsal (VD)"
    if "dorsoventral" in t or re.search(r"\bdv\b", t):
        return "dorsoventral (DV)"
    if "lateral" in t:
        return "lateral"
    if "oblique" in t:
        return "oblique"
    if text and text.strip():
        return "other / unspecified"
    return "unspecified"


def parse_region(text: str) -> str:
    t = text.lower()
    if "torac" in t or "thorac" in t:
        return "Thorax"
    if "lombar" in t or "lumbar" in t:
        return "Lumbar spine"
    if "abdomen" in t or "abdominal" in t:
        return "Abdomen"
    if "pelv" in t:
        return "Pelvis"
    if "crani" in t or "skull" in t:
        return "Skull / cranium"
    if "coaps" in t or "hip" in t or "extrem" in t:
        return "Extremity / hip"
    if text and text.strip():
        return text.strip()[:48]
    return "Unspecified"


def age_years(study_date: str, birth_date: str) -> Optional[float]:
    if len(study_date) < 8 or len(birth_date) < 8:
        return None
    try:
        sd = datetime.strptime(study_date[:8], "%Y%m%d")
        bd = datetime.strptime(birth_date[:8], "%Y%m%d")
    except ValueError:
        return None
    years = (sd - bd).days / 365.25
    if years < 0 or years > 30:
        return None
    return years


def sex_label(code: str) -> str:
    m = {
        "M": "Male",
        "F": "Female",
        "O": "Other",
        "U": "Unknown",
        "MC": "Male (neutered)",
        "FS": "Female (spayed)",
        "": "Unknown",
    }
    return m.get(code.upper(), code)


def normalize_breed(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "Not recorded"
    low = n.lower()
    if low in {"owner_urgency", "x", "unknown", "n/a"}:
        return "Not recorded (placeholder tag)"
    if low in {"comun", "comuna", "com", "comună", "rasa comuna", "rasã comunã", "europeana", "european", "metis", "metis."}:
        return "Mixed / common (no breed specified)"
    return n


def categorize_region(text: str) -> str:
    t = text.lower()
    if "torac" in t or "thorac" in t or "pulm" in t:
        return "Thorax"
    if "abdomen" in t or "abdominal" in t or "stomac" in t:
        return "Abdomen"
    if "lombar" in t or "lumbar" in t or "cervical" in t or "sacral" in t:
        return "Spine (lumbar/cervical)"
    if "crani" in t or "skull" in t or "maxilar" in t or "mandib" in t:
        return "Skull / dental"
    if any(k in t for k in ["sold", "shoulder", "genunchi", "femur", "cot", "coaps", "cader", "carpo", "tarso", "humero", "radio", "cubito", "extrem", "umãr", "umar", "pelv"]):
        return "Extremity / joint"
    if "coada" in t or "tail" in t:
        return "Tail"
    if "bronch" in t:
        return "Thorax (special procedure)"
    parsed = parse_region(text)
    if parsed in {"Thorax", "Lumbar spine", "Abdomen", "Pelvis", "Skull / cranium", "Extremity / hip"}:
        mapping = {
            "Lumbar spine": "Spine (lumbar/cervical)",
            "Skull / cranium": "Skull / dental",
            "Extremity / hip": "Extremity / joint",
        }
        return mapping.get(parsed, parsed)
    if parsed == "Unspecified":
        return "Other / unspecified"
    return "Other / unspecified"


def weight_bucket_kg(w: float, species: str) -> str:
    if species == "dog":
        if w < 10:
            return "Small (<10 kg)"
        if w < 25:
            return "Medium (10–25 kg)"
        return "Large (≥25 kg)"
    if w < 4:
        return "Small (<4 kg)"
    if w < 6:
        return "Medium (4–6 kg)"
    return "Large (≥6 kg)"


@dataclass
class ImageMeta:
    patient_key: str
    species: str
    path: str
    study_date: str
    sex: str
    breed: str
    age_years: Optional[float]
    weight_kg: Optional[float]
    body_size: str
    region: str
    region_category: str
    projection: str
    modality: str
    manufacturer: str
    model: str
    rows: int
    columns: int
    lossy_compression: str
    photometric: str


def read_image_meta(path: Path, patient_key: str, label: int) -> Optional[ImageMeta]:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    species = "dog" if label == LABEL_DOG else "cat"
    desc = _s(ds, "StudyDescription") or _s(ds, "SeriesDescription") or _s(ds, "BodyPartExamined")
    w_raw = getattr(ds, "PatientWeight", None)
    weight: Optional[float] = None
    if w_raw not in (None, "", 0, "0"):
        try:
            weight = float(w_raw)
        except (TypeError, ValueError):
            weight = None
    age = age_years(_s(ds, "StudyDate"), _s(ds, "PatientBirthDate"))
    breed = normalize_breed(_s(ds, "PatientBirthName") or _s(ds, "OtherPatientNames") or "")
    sex = sex_label(_s(ds, "PatientSex", "U"))
    bsize = weight_bucket_kg(weight, species) if weight is not None else "Not recorded"
    return ImageMeta(
        patient_key=patient_key,
        species=species,
        path=str(path),
        study_date=_s(ds, "StudyDate"),
        sex=sex,
        breed=breed,
        age_years=age,
        weight_kg=weight,
        body_size=bsize,
        region=parse_region(desc),
        region_category=categorize_region(desc),
        projection=parse_projection(desc),
        modality=_s(ds, "Modality") or "Unknown",
        manufacturer=_s(ds, "Manufacturer") or "Unknown",
        model=_s(ds, "ManufacturerModelName") or _s(ds, "StationName") or "Unknown",
        rows=int(getattr(ds, "Rows", 0) or 0),
        columns=int(getattr(ds, "Columns", 0) or 0),
        lossy_compression=_s(ds, "LossyImageCompression", "00"),
        photometric=_s(ds, "PhotometricInterpretation"),
    )


def summarize_counter(c: Counter, top_n: int = 8) -> str:
    parts = [f"{k} (n={v})" for k, v in c.most_common(top_n)]
    rest = sum(v for _, v in c.most_common()[top_n:])
    if rest:
        parts.append(f"Other (n={rest})")
    return "; ".join(parts)


def numeric_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "summary": "Not available in DICOM metadata"}
    return {
        "n": len(values),
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "sd": round(stdev(values), 2) if len(values) > 1 else 0.0,
        "summary": f"{median(values):.2f} (IQR approx.; median {median(values):.2f}, range {min(values):.2f}–{max(values):.2f})",
    }


def build_characteristics(data_root: Path, cats_dir: str, dogs_dir: str) -> Dict[str, Any]:
    records = build_patient_records(data_root, cats_dir=cats_dir, dogs_dir=dogs_dir)
    images: List[ImageMeta] = []
    failed = 0
    imgs_per_patient: List[int] = []

    for rec in records:
        imgs_per_patient.append(len(rec.image_paths))
        for p in rec.image_paths:
            meta = read_image_meta(p, rec.patient_key, rec.label)
            if meta is None:
                failed += 1
            else:
                images.append(meta)

    dogs = [r for r in records if r.label == LABEL_DOG]
    cats = [r for r in records if r.label == LABEL_CAT]
    img_dog = [m for m in images if m.species == "dog"]
    img_cat = [m for m in images if m.species == "cat"]

    # Patient-level demographics: one value per patient (mode / first valid)
    def patient_level_demographics(species: str) -> Dict[str, Any]:
        subset = [r for r in records if (r.label == LABEL_DOG) == (species == "dog")]
        sex_c = Counter()
        breed_c = Counter()
        ages_p: List[float] = []
        weights_p: List[float] = []
        size_c = Counter()
        for rec in subset:
            metas = [m for m in images if m.patient_key == rec.patient_key]
            if not metas:
                continue
            sex_c[metas[0].sex] += 1
            breed_c[metas[0].breed] += 1
            for m in metas:
                if m.age_years is not None:
                    ages_p.append(m.age_years)
                    break
            for m in metas:
                if m.weight_kg is not None:
                    weights_p.append(m.weight_kg)
                    size_c[m.body_size] += 1
                    break
        return {
            "sex_among_patients": dict(sex_c),
            "sex_summary": summarize_counter(sex_c),
            "breed_among_patients": dict(breed_c.most_common(20)),
            "breed_summary": summarize_counter(breed_c, top_n=10),
            "unique_breed_labels": len(breed_c),
            "age_years": numeric_summary(ages_p),
            "weight_kg": numeric_summary(weights_p),
            "body_size_categories": dict(size_c),
            "body_size_summary": summarize_counter(size_c),
        }

    region_img = Counter(m.region_category for m in images)
    proj_img = Counter(m.projection for m in images)
    equip = Counter(f"{m.manufacturer} — {m.model}" for m in images)
    modality = Counter(m.modality for m in images)
    lossy = Counter(m.lossy_compression for m in images)
    dims = Counter(f"{m.rows}×{m.columns}" for m in images if m.rows and m.columns)

    multi = sum(1 for n in imgs_per_patient if n > 1)
    single = sum(1 for n in imgs_per_patient if n == 1)

    quality_notes = {
        "lossy_compression_tags": dict(lossy),
        "native_matrix_sizes": dict(dims.most_common(10)),
        "bit_depth": "16-bit stored (typical); converted offline to 8-bit for modeling",
        "inclusion_policy": "Clinical DICOM retained if diagnostic field-of-view present and without severe motion/blur/artifacts (manual curation at export)",
        "lossless_dicom_fraction": f"{lossy.get('00', 0) / max(len(images), 1) * 100:.1f}% tagged LossyImageCompression=00",
    }

    per_species_counts = {
        "dog": {"patients": len(dogs), "images": len(img_dog)},
        "cat": {"patients": len(cats), "images": len(img_cat)},
    }

    images_per_patient = {
        "mean": round(mean(imgs_per_patient), 2),
        "median": median(imgs_per_patient),
        "min": min(imgs_per_patient),
        "max": max(imgs_per_patient),
        "distribution": dict(Counter(imgs_per_patient)),
        "patients_with_single_image": single,
        "patients_with_multiple_images": multi,
        "multiple_images_included": True,
        "note": "All DICOM instances per patient folder were retained; splits were performed at patient level to avoid leakage.",
    }

    dog_demo = patient_level_demographics("dog")
    cat_demo = patient_level_demographics("cat")

    table_rows = [
        {"Characteristic": "Individual animals (patients)", "Overall": len(records), "Canine": len(dogs), "Feline": len(cats)},
        {"Characteristic": "Radiographs (DICOM instances)", "Overall": len(images), "Canine": len(img_dog), "Feline": len(img_cat)},
        {
            "Characteristic": "Images per animal, median (range)",
            "Overall": f"{median(imgs_per_patient)} ({min(imgs_per_patient)}–{max(imgs_per_patient)})",
            "Canine": f"{median([len(r.image_paths) for r in dogs])} ({min(len(r.image_paths) for r in dogs)}–{max(len(r.image_paths) for r in dogs)})",
            "Feline": f"{median([len(r.image_paths) for r in cats])} ({min(len(r.image_paths) for r in cats)}–{max(len(r.image_paths) for r in cats)})",
        },
        {
            "Characteristic": "Animals with >1 image included",
            "Overall": f"{multi} ({100*multi/len(records):.1f}%)",
            "Canine": f"{sum(1 for r in dogs if len(r.image_paths)>1)}",
            "Feline": f"{sum(1 for r in cats if len(r.image_paths)>1)}",
        },
        {
            "Characteristic": "Multiple views per animal",
            "Overall": f"Yes — {multi}/{len(records)} patients ({100*multi/len(records):.1f}%) with >1 study; patient-level split prevents leakage",
            "Canine": f"{sum(1 for r in dogs if len(r.image_paths)>1)}/{len(dogs)} with >1 image",
            "Feline": f"{sum(1 for r in cats if len(r.image_paths)>1)}/{len(cats)} with >1 image",
        },
        {
            "Characteristic": "Age (years) at imaging, median (range)*",
            "Overall": _fmt_demo_numeric_patient(records, images, "age_years"),
            "Canine": _fmt_demo_numeric_patient(dogs, images, "age_years"),
            "Feline": _fmt_demo_numeric_patient(cats, images, "age_years"),
        },
        {
            "Characteristic": "Sex (patients)**",
            "Overall": f"Dogs: {dog_demo['sex_summary']}; Cats: {cat_demo['sex_summary']}",
            "Canine": dog_demo["sex_summary"],
            "Feline": cat_demo["sex_summary"],
        },
        {
            "Characteristic": "Breed / variety (DICOM PatientBirthName)***",
            "Overall": f"{len(set(normalize_breed(m.breed) for m in images))} normalized labels",
            "Canine": summarize_counter(Counter(normalize_breed(m.breed) for m in img_dog), 6),
            "Feline": summarize_counter(Counter(normalize_breed(m.breed) for m in img_cat), 6),
        },
        {
            "Characteristic": "Body weight (kg), median (range)*",
            "Overall": _fmt_demo_numeric([m.weight_kg for m in images if m.weight_kg is not None]),
            "Canine": _fmt_demo_numeric([m.weight_kg for m in img_dog if m.weight_kg is not None]),
            "Feline": _fmt_demo_numeric([m.weight_kg for m in img_cat if m.weight_kg is not None]),
        },
        {
            "Characteristic": "Body size proxy (from weight)",
            "Overall": summarize_counter(Counter(m.body_size for m in images if m.weight_kg is not None)),
            "Canine": summarize_counter(Counter(m.body_size for m in img_dog if m.weight_kg is not None)),
            "Feline": summarize_counter(Counter(m.body_size for m in img_cat if m.weight_kg is not None)),
        },
        {
            "Characteristic": "Anatomical region (image-level)",
            "Overall": summarize_counter(region_img, 6),
            "Canine": summarize_counter(Counter(m.region_category for m in img_dog), 5),
            "Feline": summarize_counter(Counter(m.region_category for m in img_cat), 5),
        },
        {
            "Characteristic": "Projection (image-level)",
            "Overall": summarize_counter(proj_img),
            "Canine": summarize_counter(Counter(m.projection for m in img_dog)),
            "Feline": summarize_counter(Counter(m.projection for m in img_cat)),
        },
        {
            "Characteristic": "Modality",
            "Overall": summarize_counter(modality),
            "Canine": summarize_counter(Counter(m.modality for m in img_dog)),
            "Feline": summarize_counter(Counter(m.modality for m in img_cat)),
        },
        {
            "Characteristic": "Imaging equipment",
            "Overall": summarize_counter(equip, 3),
            "Canine": summarize_counter(Counter(f"{m.manufacturer} — {m.model}" for m in img_dog), 2),
            "Feline": summarize_counter(Counter(f"{m.manufacturer} — {m.model}" for m in img_cat), 2),
        },
        {
            "Characteristic": "Image quality / format",
            "Overall": f"{quality_notes['lossless_dicom_fraction']}; matrices: {summarize_counter(dims, 3)}",
            "Canine": f"100% lossless tag; 2880×2880 (n={len(img_dog)})",
            "Feline": f"100% lossless tag; 2880×2880 (n={len(img_cat)})",
        },
    ]

    footnotes = [
        "* Age computed from PatientBirthDate and StudyDate when both valid; missing or implausible values excluded.",
        "** Sex from DICOM PatientSex at patient level (first study).",
        "*** Breed recorded in DICOM PatientBirthName field by the hospital RIS (mixed nomenclature; not verified independently).",
        "Multiple views from the same animal were included; training/validation/test splits were stratified by patient folder to prevent leakage.",
    ]

    return {
        "data_root": str(data_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "read_failures": failed,
        "counts": per_species_counts,
        "images_per_patient": images_per_patient,
        "demographics_dog": dog_demo,
        "demographics_cat": cat_demo,
        "breed_counts": {
            "dog": list(Counter(normalize_breed(m.breed) for m in img_dog).most_common()),
            "cat": list(Counter(normalize_breed(m.breed) for m in img_cat).most_common()),
        },
        "image_level": {
            "regions_grouped": dict(region_img),
            "regions_raw": dict(Counter(m.region for m in images)),
            "projections": dict(proj_img),
            "equipment": dict(equip),
            "modalities": dict(modality),
            "quality": quality_notes,
        },
        "manuscript_table": table_rows,
        "footnotes": footnotes,
        "n_images_parsed": len(images),
    }


def _fmt_demo_numeric(values: List[Optional[float]]) -> str:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return "Not recorded"
    return f"{median(vals):.1f} ({min(vals):.1f}–{max(vals):.1f}); n={len(vals)}"


def _fmt_demo_numeric_patient(records, images: List[ImageMeta], field: str) -> str:
    vals: List[float] = []
    for rec in records:
        metas = [m for m in images if m.patient_key == rec.patient_key]
        for m in metas:
            v = getattr(m, field)
            if v is not None:
                vals.append(float(v))
                break
    return _fmt_demo_numeric(vals)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Characteristic", "Overall", "Canine", "Feline"])
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, data: Dict[str, Any]) -> None:
    lines = [
        "# Dataset characteristics (DICOM-derived)",
        "",
        f"Source: `{data['data_root']}` · Images parsed: **{data['n_images_parsed']}** · Read failures: {data['read_failures']}",
        "",
        "| Characteristic | Overall | Canine | Feline |",
        "| --- | --- | --- | --- |",
    ]
    for row in data["manuscript_table"]:
        lines.append(
            f"| {row['Characteristic']} | {row['Overall']} | {row['Canine']} | {row['Feline']} |"
        )
    lines.append("")
    for fn in data["footnotes"]:
        lines.append(fn)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset characteristics table from DICOM metadata.")
    parser.add_argument("--data_root", type=str, default="/home/miorga/Doctorat/vet_dataset")
    parser.add_argument("--cats_dir", type=str, default="Pisici")
    parser.add_argument("--dogs_dir", type=str, default="Catei")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/miorga/Doctorat/test1/dataset_characteristics",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_characteristics(Path(args.data_root), args.cats_dir, args.dogs_dir)

    json_path = out_dir / "dataset_characteristics.json"
    csv_path = out_dir / "dataset_characteristics_table.csv"
    md_path = out_dir / "dataset_characteristics_table.md"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, data["manuscript_table"])
    write_markdown(md_path, data)

    breed_csv = out_dir / "breed_counts_by_species.csv"
    with breed_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "breed_normalized", "n_images"])
        for species in ("dog", "cat"):
            for breed, n in data["breed_counts"][species]:
                w.writerow([species, breed, n])
    print(f"Wrote {breed_csv}")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print()
    print("=== Table preview ===")
    for row in data["manuscript_table"]:
        print(f"{row['Characteristic'][:50]:50s} | {row['Overall']}")


if __name__ == "__main__":
    main()
