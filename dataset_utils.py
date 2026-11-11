"""Discover DICOM files and patient-level grouping for vet_dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from sklearn.model_selection import train_test_split


LABEL_DOG = 0
LABEL_CAT = 1


def is_dicom_file(path: Path) -> bool:
    if not path.is_file():
        return False
    # Media storage directory index (no pixel data).
    if path.name.upper() == "DICOMDIR":
        return False
    if path.suffix.lower() == ".dcm":
        return True
    if path.suffix != "":
        return False
    try:
        with path.open("rb") as f:
            header = f.read(132)
    except OSError:
        return False
    return len(header) >= 132 and header[128:132] == b"DICM"


def iter_patient_dirs(class_dir: Path) -> Iterator[Path]:
    if not class_dir.is_dir():
        return
    for entry in sorted(class_dir.iterdir()):
        if entry.is_dir():
            yield entry


def collect_patient_dicoms(patient_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in sorted(patient_dir.rglob("*")):
        if is_dicom_file(p):
            paths.append(p)
    return paths


@dataclass(frozen=True)
class PatientRecord:
    patient_key: str
    label: int
    image_paths: Tuple[Path, ...]


def build_patient_records(
    data_root: Path,
    cats_dir: str = "Pisici",
    dogs_dir: str = "Catei",
) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    for label, species_dir in ((LABEL_CAT, cats_dir), (LABEL_DOG, dogs_dir)):
        class_path = data_root / species_dir
        if not class_path.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_path}")
        for patient_dir in iter_patient_dirs(class_path):
            images = collect_patient_dicoms(patient_dir)
            if not images:
                continue
            patient_key = f"{species_dir}/{patient_dir.name}"
            records.append(
                PatientRecord(
                    patient_key=patient_key,
                    label=label,
                    image_paths=tuple(images),
                )
            )
    return records


def flatten_patient_records(records: List[PatientRecord]) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for rec in records:
        for p in rec.image_paths:
            samples.append((p, rec.label))
    return samples


def patient_key_for_image(image_path: Path, data_root: Path) -> str:
    try:
        rel = image_path.relative_to(data_root)
    except ValueError:
        return str(image_path.parent)
    parts = rel.parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return str(rel.parent)


@dataclass
class Split:
    train: List[Tuple[str, int]]
    val: List[Tuple[str, int]]
    test: List[Tuple[str, int]]
    split_level: str = "patient"
    patients: Dict[str, Any] | None = None


def stratified_split_by_patient(
    records: List[PatientRecord],
    seed: int,
    test_size: float,
    val_size: float,
) -> Split:
    if len(records) == 0:
        raise ValueError("No patient records to split.")

    keys = [r.patient_key for r in records]
    labels = [r.label for r in records]
    path_by_key = {r.patient_key: [str(p) for p in r.image_paths] for r in records}

    train_val_keys, test_keys, train_val_labels, test_labels = train_test_split(
        keys,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    val_size_within_train_val = val_size / (1.0 - test_size)
    train_keys, val_keys, train_labels, val_labels = train_test_split(
        train_val_keys,
        train_val_labels,
        test_size=val_size_within_train_val,
        random_state=seed,
        stratify=train_val_labels,
    )

    def expand_patient_keys(patient_keys: List[str], patient_labels: List[int]) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        label_by_key = dict(zip(patient_keys, patient_labels))
        for pk in patient_keys:
            y = int(label_by_key[pk])
            for img in path_by_key[pk]:
                out.append((img, y))
        return out

    patient_meta = {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys,
    }

    return Split(
        train=expand_patient_keys(train_keys, train_labels),
        val=expand_patient_keys(val_keys, val_labels),
        test=expand_patient_keys(test_keys, test_labels),
        split_level="patient",
        patients=patient_meta,
    )


def stratified_split_by_image(
    samples: List[Tuple[Path, int]],
    seed: int,
    test_size: float,
    val_size: float,
) -> Split:
    labels = [y for _, y in samples]
    paths = [str(p) for p, _ in samples]

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    val_size_within_train_val = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size_within_train_val,
        random_state=seed,
        stratify=train_val_labels,
    )

    def pack(ps: List[str], ys: List[int]) -> List[Tuple[str, int]]:
        return [(p, int(y)) for p, y in zip(ps, ys)]

    return Split(
        train=pack(train_paths, train_labels),
        val=pack(val_paths, val_labels),
        test=pack(test_paths, test_labels),
        split_level="image",
        patients=None,
    )


def compute_dataset_statistics(data_root: Path, cats_dir: str = "Pisici", dogs_dir: str = "Catei") -> Dict[str, Any]:
    records = build_patient_records(data_root, cats_dir=cats_dir, dogs_dir=dogs_dir)
    by_class: Dict[str, Dict[str, int]] = {
        "dog": {"patients": 0, "dicom_images": 0},
        "cat": {"patients": 0, "dicom_images": 0},
    }
    for rec in records:
        name = "cat" if rec.label == LABEL_CAT else "dog"
        by_class[name]["patients"] += 1
        by_class[name]["dicom_images"] += len(rec.image_paths)

    total_patients = sum(v["patients"] for v in by_class.values())
    total_images = sum(v["dicom_images"] for v in by_class.values())

    return {
        "data_root": str(data_root),
        "cats_dir": cats_dir,
        "dogs_dir": dogs_dir,
        "by_class": by_class,
        "total_patients": total_patients,
        "total_dicom_images": total_images,
    }


def split_to_json_dict(split: Split) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "split_level": split.split_level,
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }
    if split.patients is not None:
        payload["patients"] = split.patients
    return payload


def resolve_dicom_path(path_str: str, data_root: Path) -> Path:
    """Absolute paths are kept; relative paths are resolved under data_root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (data_root / p).resolve()


def load_split_from_json(raw: Dict[str, Any]) -> Split:
    return Split(
        train=[tuple(x) for x in raw["train"]],  # type: ignore[misc]
        val=[tuple(x) for x in raw["val"]],  # type: ignore[misc]
        test=[tuple(x) for x in raw["test"]],  # type: ignore[misc]
        split_level=str(raw.get("split_level", "image")),
        patients=raw.get("patients"),
    )


def splits_file_is_compatible(
    splits_path: Path,
    sample_paths: List[Path],
    data_root: Path,
) -> bool:
    if not splits_path.exists():
        return False
    try:
        loaded = json.loads(splits_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    all_paths = [p for split_name in ("train", "val", "test") for p, _ in loaded.get(split_name, [])]
    if not all_paths:
        return False
    sample_set = {str(p.resolve()) for p in sample_paths}
    for p in all_paths:
        if str(resolve_dicom_path(str(p), data_root)) not in sample_set:
            return False
    return True
