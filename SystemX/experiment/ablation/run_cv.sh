#!/usr/bin/env bash
_REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-${_REPO_ROOT}/.venv/bin/python3}"
if [ ! -f "$PYTHON" ]; then PYTHON="python3"; fi

set -euo pipefail

ANNOTATIONS_DIR="data/jetbrains_dataset/annotations"
RAW_DIR="data/jetbrains_dataset/notebooks"
OUTPUT_ROOT="output/cv"
RESULTS_PATH="output/results/cv_results.json"
FIGURES_DIR="output/figures"
CONFIG_PATH="systemx-jupyter/models/config/default_reversed.yaml"
FOLDS=5
SEED=42
EPOCHS_HGT=80
EPOCHS_MLP=50
HGT_FEATURES=""
FORCE=""
SKIP_TRAINING=""
SKIP_EVALUATION=""
VARIANTS_FLAG=""

_SCRIPT_DIR="SystemX/experiment/ablation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --annotations_dir) ANNOTATIONS_DIR="$2"; shift 2 ;;
    --raw_dir)         RAW_DIR="$2"; shift 2 ;;
    --output_root)     OUTPUT_ROOT="$2"; shift 2 ;;
    --results_path)    RESULTS_PATH="$2"; shift 2 ;;
    --figures_dir)     FIGURES_DIR="$2"; shift 2 ;;
    --folds)           FOLDS="$2"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    --epochs_hgt)      EPOCHS_HGT="$2"; shift 2 ;;
    --epochs_mlp)      EPOCHS_MLP="$2"; shift 2 ;;
    --hgt_features)    HGT_FEATURES="$2"; shift 2 ;;
    --variants)        shift; VARIANTS_FLAG="--variants"; while [[ $# -gt 0 && "$1" != --* ]]; do VARIANTS_FLAG="$VARIANTS_FLAG $1"; shift; done ;;
    --force)           FORCE="--force"; shift ;;
    --skip_training)   SKIP_TRAINING="--skip_training"; shift ;;
    --skip_evaluation) SKIP_EVALUATION="--skip_evaluation"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  SystemX ${FOLDS}-fold Cross-Validation Pipeline"
echo "============================================================"
echo "  annotations_dir : ${ANNOTATIONS_DIR}"
echo "  raw_dir         : ${RAW_DIR}"
echo "  output_root     : ${OUTPUT_ROOT}"
echo "  results_path    : ${RESULTS_PATH}"
echo "  figures_dir     : ${FIGURES_DIR}"
echo "  folds / seed    : ${FOLDS} / ${SEED}"
echo "  epochs_hgt/mlp  : ${EPOCHS_HGT} / ${EPOCHS_MLP}"
echo "============================================================"

echo ""
echo ">>> STEP 1+2: Cross-validated training & held-out evaluation"
"${PYTHON}" "${_SCRIPT_DIR}/run_cv.py" \
  --annotations_dir "${ANNOTATIONS_DIR}" \
  --raw_dir         "${RAW_DIR}" \
  --config_path     "${CONFIG_PATH}" \
  --output_root     "${OUTPUT_ROOT}" \
  --results_path    "${RESULTS_PATH}" \
  --folds           "${FOLDS}" \
  --seed            "${SEED}" \
  --epochs_hgt      "${EPOCHS_HGT}" \
  --epochs_mlp      "${EPOCHS_MLP}" \
  ${HGT_FEATURES:+--hgt_features "${HGT_FEATURES}"} \
  ${VARIANTS_FLAG} ${FORCE} ${SKIP_TRAINING} ${SKIP_EVALUATION}

echo ""
echo ">>> STEP 3: Generating paper figures"
"${PYTHON}" "${_SCRIPT_DIR}/plot_results.py" \
  --results_path "${RESULTS_PATH}" \
  --output_dir   "${FIGURES_DIR}"

echo ""
echo "============================================================"
echo "  CV pipeline complete!"
echo "  Per-fold     : ${OUTPUT_ROOT}/fold_*/"
echo "  Aggregated   : ${RESULTS_PATH}"
echo "  Figures      : ${FIGURES_DIR}/"
echo "============================================================"
