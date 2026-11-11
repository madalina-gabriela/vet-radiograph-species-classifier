"""DICOM dataset with offline preprocess mode (default / CLAHE)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dicom_preprocess import CACHE_VERSION, dicom_to_uint8, load_uint8_for_mode


class VetDicomDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        cache_dir: Path,
        transform: Optional[Any] = None,
        force_recache: bool = False,
        offline_preprocess: str = "default",
    ) -> None:
        self.samples = samples
        self.cache_dir = cache_dir
        self.transform = transform
        self.force_recache = force_recache
        self.offline_preprocess = offline_preprocess
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, dcm_path: Path) -> Path:
        key = f"{CACHE_VERSION}:{self.offline_preprocess}:{dcm_path}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()  # nosec
        return self.cache_dir / f"{h}.npy"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        dcm_path, label = self.samples[idx]
        cache_path = self._cache_path(dcm_path)
        if (not self.force_recache) and cache_path.exists():
            arr_uint8 = np.load(str(cache_path))
        else:
            arr_uint8 = load_uint8_for_mode(dcm_path, self.offline_preprocess)
            np.save(str(cache_path), arr_uint8)
        img = Image.fromarray(arr_uint8, mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
