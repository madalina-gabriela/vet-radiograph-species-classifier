#!/usr/bin/env bash
# Antrenare + evaluare cap-coada pe vet_dataset (split la nivel de pacient).
set -euo pipefail
ROOT="/home/miorga/Doctorat"
cd "$ROOT/vet1_catdog"
PY="$ROOT/venv/bin/python"

echo "=== 1/2 Statistici set de date ==="
"$PY" dataset_stats.py --output_json "$ROOT/vet1_catdog/dataset_statistics.json"

echo ""
echo "=== 2/2 Antrenare ResNet18 + metrici pe test ==="
"$PY" train_catdog.py \
  --data_root "$ROOT/vet_dataset" \
  --split_level patient \
  --splits_file "$ROOT/vet1_catdog/splits_vet_dataset_patient.json" \
  --output_dir "$ROOT/Results" \
  --cache_dir "$ROOT/vet1_catdog/cache_dicom_uint8_vet_dataset_v2" \
  "$@"

echo ""
echo "=== Rezultate ==="
"$PY" - <<'PY'
import json
from pathlib import Path
p = Path("/home/miorga/Doctorat/Results/metrics.json")
if p.exists():
    data = json.loads(p.read_text())
    mi = data.get("metrics_image_level") or data.get("metrics", {})
    mp = data.get("metrics_patient_level", {})
    print(f"Test accuracy (imagine):  {mi.get('accuracy', 0):.4f} ({100*mi.get('accuracy', 0):.2f}%)")
    if mp:
        print(f"Test accuracy (pacient):  {mp.get('accuracy', 0):.4f} ({100*mp.get('accuracy', 0):.2f}%)")
    print(f"Balanced (imagine):       {mi.get('balanced_accuracy')}")
    print(f"ROC-AUC (imagine):        {mi.get('roc_auc_cat_vs_dog')}")
    print(f"Fișier complet: {p}")
else:
    print("metrics.json lipsește — antrenarea nu s-a terminat cu succes.")
PY
