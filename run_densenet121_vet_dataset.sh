#!/usr/bin/env bash
# DenseNet-121 on vet_dataset — same patient split as ResNet-18 (test1).
set -euo pipefail
ROOT="/home/miorga/Doctorat"
cd "$ROOT/vet1_catdog"
PY="$ROOT/venv/bin/python"
OUT="${1:-$ROOT/test_densenet121/run_seed_42}"
SEED="${SEED:-42}"

mkdir -p "$OUT"

echo "=== DenseNet-121 training (output: $OUT, seed=$SEED) ==="
"$PY" train_densenet121.py \
  --data_root "$ROOT/vet_dataset" \
  --split_level patient \
  --splits_file "$ROOT/vet1_catdog/splits_vet_dataset_patient.json" \
  --cache_dir "$ROOT/vet1_catdog/cache_dicom_uint8_vet_dataset_v2" \
  --output_dir "$OUT" \
  --seed "$SEED" \
  --epochs_fc_only 5 \
  --epochs_finetune 8 \
  --fc_dropout 0 \
  --batch_size 16 \
  "${@:2}"

echo ""
echo "=== Test summary ==="
"$PY" - <<PY
import json
from pathlib import Path
p = Path("$OUT/metrics.json")
if not p.exists():
    raise SystemExit("metrics.json missing")
data = json.loads(p.read_text())
mi = data.get("metrics_image_level") or data.get("metrics", {})
mp = data.get("metrics_patient_level", {})
print(f"Architecture: {data.get('architecture', 'densenet121')}")
print(f"Test accuracy (image):  {100*mi.get('accuracy', 0):.2f}%")
print(f"Test balanced (image):  {100*mi.get('balanced_accuracy', 0):.2f}%")
print(f"Test accuracy (patient): {100*mp.get('accuracy', 0):.2f}%")
print(f"ROC-AUC: {mi.get('roc_auc_cat_vs_dog')}")
print(f"Best val acc: {100*data.get('best_val_acc', 0):.2f}%")
print(f"Metrics: {p}")
PY
