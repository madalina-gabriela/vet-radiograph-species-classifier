"""Discover DICOM files and patient-level grouping for vet_dataset."""

from __future__ import annotations

import hashlib
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
        return anonymize_patient_key(f"{parts[0]}/{parts[1]}")
    return str(rel.parent)


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def anonymize_patient_key(patient_key: str) -> str:
    """Map e.g. Catei/Patient 27 -> Catei/pt_<hash(species/patient_folder)>."""
    parts = patient_key.split("/", 1)
    if len(parts) != 2:
        return patient_key
    species, folder = parts
    return f"{species}/pt_{_sha16(f'{species}/{folder}')}"


def anonymize_split_path(rel_path: str) -> str:
    """Map relative DICOM path to species/pt_*/im_* without patient numbers or UIDs."""
    rel = rel_path.replace("\\", "/")
    parts = rel.split("/")
    if len(parts) < 3:
        return rel
    species, patient_folder, _instance = parts[0], parts[1], parts[2]
    pt = f"pt_{_sha16(f'{species}/{patient_folder}')}"
    im = f"im_{_sha16(rel)}"
    return f"{species}/{pt}/{im}"


def _relative_posix(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root.resolve()).as_posix()


_PATH_LOOKUP_CACHE: Dict[Tuple[str, str, str], Dict[str, Path]] = {}


def build_path_lookup(
    data_root: Path,
    cats_dir: str = "Pisici",
    dogs_dir: str = "Catei",
) -> Dict[str, Path]:
    """Keys: raw relative, anonymized relative, and resolved absolute path strings."""
    root = data_root.resolve()
    cache_key = (str(root), cats_dir, dogs_dir)
    if cache_key in _PATH_LOOKUP_CACHE:
        return _PATH_LOOKUP_CACHE[cache_key]

    lookup: Dict[str, Path] = {}
    for path, _label in flatten_patient_records(
        build_patient_records(root, cats_dir=cats_dir, dogs_dir=dogs_dir)
    ):
        resolved = path.resolve()
        rel = _relative_posix(resolved, root)
        anon = anonymize_split_path(rel)
        for key in (rel, anon, str(resolved)):
            lookup[key] = resolved

    _PATH_LOOKUP_CACHE[cache_key] = lookup
    return lookup


def clear_path_lookup_cache() -> None:
    _PATH_LOOKUP_CACHE.clear()


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


def split_to_json_dict(
    split: Split,
    data_root: Path | None = None,
    *,
    anonymize_paths: bool = True,
) -> Dict[str, Any]:
    def encode(entries: List[Tuple[str, int]]) -> List[List[int | str]]:
        out: List[List[int | str]] = []
        for path_str, label in entries:
            if data_root is not None and anonymize_paths:
                out.append([_split_path_for_json(str(path_str), data_root), int(label)])
            else:
                out.append([path_str, int(label)])
        return out

    payload: Dict[str, Any] = {
        "split_level": split.split_level,
        "train": encode(split.train),
        "val": encode(split.val),
        "test": encode(split.test),
    }
    if split.patients is not None:
        if data_root is not None and anonymize_paths:
            payload["patients"] = {
                split_name: [anonymize_patient_key(pk) for pk in keys]
                for split_name, keys in split.patients.items()
            }
        else:
            payload["patients"] = split.patients
    if data_root is not None and anonymize_paths:
        payload["path_format"] = "anonymized_relative_to_data_root"
        payload["anonymization_note"] = (
            "pt_* and im_* are SHA-256 truncations of species/patient-folder and full relative path; "
            "on-disk folders are unchanged. Resolved at runtime via build_path_lookup()."
        )
    return payload


def _split_path_for_json(path_str: str, data_root: Path) -> str:
    p = Path(path_str)
    if p.is_absolute():
        rel = _relative_posix(p.resolve(), data_root.resolve())
    else:
        rel = path_str.replace("\\", "/")
    parts = rel.split("/")
    if len(parts) >= 2 and parts[1].startswith("pt_"):
        return rel
    return anonymize_split_path(rel)


def resolve_dicom_path(
    path_str: str,
    data_root: Path,
    cats_dir: str = "Pisici",
    dogs_dir: str = "Catei",
) -> Path:
    """Resolve plain, absolute, or anonymized split paths against data_root."""
    raw = str(path_str).replace("\\", "/")
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()

    direct = (data_root.resolve() / p).resolve()
    if direct.is_file():
        return direct

    lookup = build_path_lookup(data_root, cats_dir=cats_dir, dogs_dir=dogs_dir)
    if raw in lookup:
        return lookup[raw]
    parts = raw.split("/")
    if len(parts) >= 2 and not parts[1].startswith("pt_"):
        anon = anonymize_split_path(raw)
        if anon in lookup:
            return lookup[anon]

    raise FileNotFoundError(f"Split path not found under data_root: {raw!r}")


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
    cats_dir: str = "Pisici",
    dogs_dir: str = "Catei",
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
        try:
            resolved = resolve_dicom_path(str(p), data_root, cats_dir=cats_dir, dogs_dir=dogs_dir)
        except FileNotFoundError:
            return False
        if str(resolved) not in sample_set:
            return False
    return True
