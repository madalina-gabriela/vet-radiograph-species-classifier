#!/usr/bin/env bash
# Run ONE experiment (M0–M10): both LRs, then select best on validation macro-F1.
#
# Usage:
#   ./run_experiment.sh M0              # seed 42
#   ./run_experiment.sh M1 123          # seed 123
#
set -euo pipefail
ROOT="/home/miorga/Doctorat"
cd "$ROOT/vet1_catdog"
PY="$ROOT/venv/bin/python"

EXP="${1:?Usage: run_experiment.sh M0 [seed]}"
SEED="${2:-42}"
OUT="$ROOT/test_comparison/$EXP"

if [[ "$EXP" == "M6" ]]; then
  if ! "$PY" -c "import cv2" 2>/dev/null; then
    echo "M6 needs OpenCV: $PY -m pip install opencv-python-headless"
    exit 1
  fi
fi

mkdir -p "$OUT"
"$PY" -c "from experiment_registry import get_experiment; import json; print(json.dumps(get_experiment('$EXP'), indent=2))" \
  > "$OUT/experiment_config.json"

echo "=== $EXP (seed=$SEED) — LR grid 1e-4 and 3e-4 ==="
for LR in 1e-4 3e-4; do
  RUN="$OUT/lr_${LR}_seed_${SEED}"
  echo "--- lr=$LR -> $RUN ---"
  "$PY" train_vet_classifier.py \
    --experiment "$EXP" \
    --lr "$LR" \
    --seed "$SEED" \
    --output_dir "$RUN"
done

echo "=== Select best LR on validation macro-F1 ==="
"$PY" select_best_lr.py --dir "$OUT" --experiment "$EXP"

echo "=== Training curves (best run) ==="
BEST_METRICS="$OUT/best/metrics.json"
if [[ -f "$BEST_METRICS" ]]; then
  "$PY" plot_figure3_accuracy.py --metrics "$BEST_METRICS" \
    --output "$OUT/best/figure_validation_accuracy.png" 2>/dev/null || true
fi

echo ""
echo "Done. Primary metrics: $OUT/best/metrics.json"
echo "When you finish more experiments, plot all: $PY plot_comparison_models.py"
