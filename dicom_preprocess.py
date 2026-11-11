"""
Offline DICOM preprocessing (cached as uint8 before online augmentations).

Pipeline per image:
  1. Read pixel array
  2. Modality LUT (when present)
  3. VOI/window LUT (when present), else min–max to 8-bit
  4. Invert if PhotometricInterpretation == MONOCHROME1
  5. Clip 1st–99th percentile and scale to uint8 [0, 255]

Training-time random crops/flips/etc. are applied online in train_catdog.make_transforms.
"""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

# Bump when offline preprocessing changes (invalidates .npy cache).
CACHE_VERSION = "voi_lut_percentile_v2_monochrome1"


def dicom_to_uint8(dcm_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)

    try:
        arr = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:
        pass

    used_voi = False
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
        used_voi = True
    except Exception:
        mn, mx = float(arr.min()), float(arr.max())
        arr = (arr - mn) / (mx - mn + 1e-6) * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return _apply_monochrome1_inversion(arr, ds)

    if used_voi:
        lo = float(np.percentile(arr, 1))
        hi = float(np.percentile(arr, 99))
        if hi <= lo + 1e-6:
            lo = float(arr.min())
            hi = float(arr.max())
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-6)
        arr = (arr * 255.0).astype(np.uint8)

    return _apply_monochrome1_inversion(arr, ds)


def _apply_monochrome1_inversion(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "")
    if photometric.upper() == "MONOCHROME1":
        return (255 - arr).astype(np.uint8)
    return arr.astype(np.uint8)


def apply_clahe_uint8(arr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """CLAHE on 8-bit grayscale (requires opencv-python-headless for M6)."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise ImportError(
            "M6 (CLAHE) needs OpenCV: pip install opencv-python-headless"
        ) from e
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(arr.astype(np.uint8))


def load_uint8_for_mode(dcm_path: Path, mode: str = "default") -> np.ndarray:
    arr = dicom_to_uint8(dcm_path)
    if mode == "default":
        return arr
    if mode == "clahe":
        return apply_clahe_uint8(arr)
    raise ValueError(f"Unknown offline preprocess mode: {mode!r}")
