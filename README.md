# Veterinary radiograph species classifier (cat vs dog)

PyTorch code for patient-level train/validation/test splits, DICOM preprocessing, and comparative experiments (ResNet-18, DenseNet-121, EfficientNet-B0, CBAM) on grayscale radiographs.

**DICOM images are not included** in this repository. You must obtain data under your own institutional agreement and point `--data_root` at a folder with:

```
data_root/
  Pisici/          # cats (label 1)
    Patient <id>/<instance_uid>
  Catei/           # dogs (label 0)
    Patient <id>/<instance_uid>
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install CPU or CUDA builds of PyTorch from [pytorch.org](https://pytorch.org) if the defaults above do not match your hardware.

## Splits

`splits_vet_dataset_patient.json` is a **fixed patient-level split** (seed 42) for reproducibility. Paths are **relative to `--data_root`**.

- 592 images, 187 patients (paper cohort)
- train / val / test defined at patient level

If the split file does not match your local tree, training scripts regenerate splits (same logic) when you pass a writable `--splits_file`.

## Training (main comparison pipeline)

```bash
python train_vet_classifier.py \
  --experiment M0 \
  --data_root /path/to/vet_dataset \
  --output_dir ./outputs/M0 \
  --cache_dir ./cache_dicom_uint8
```

List experiment codes: `python train_vet_classifier.py --list_experiments`

Legacy ResNet-only script: `train_catdog.py`. DenseNet baseline: `train_densenet121.py`.

## Analysis utilities

| Script | Purpose |
|--------|---------|
| `gradcam.py` | Grad-CAM heatmaps from a checkpoint + predictions CSV |
| `compare_inference_statistics.py` | Bootstrap / McNemar across runs |
| `plot_comparison_models.py` | Bar charts from `metrics.json` |
| `analyze_test_patient_errors.py` | Cross-model patient error table |

Checkpoints (`*.pt`) are **not** shipped with the repo; train locally or request weights from the authors per journal policy.

## Ethics and data use

Radiographs were collected under third-party hospital governance. Do not redistribute DICOM or derived caches publicly. This repository contains **code and split definitions only**.

## Citation

If you use this code, cite the associated BioMedInformatics manuscript (species classification feasibility study, patient-level validation).

## License

Code is provided for research reproducibility. Add a `LICENSE` file before public release if your institution requires a specific license (e.g. MIT, Apache-2.0).
