#!/usr/bin/env bash
_REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-${_REPO_ROOT}/.venv/bin/python3}"
if [ ! -f "$PYTHON" ]; then PYTHON="python3"; fi

set -euo pipefail

ANNOTATIONS_DIR="data/jetbrains_dataset/annotations"
RAW_DIR="data/jetbrains_dataset/notebooks"
OUTPUT_DIR="output/checkpoints"
RESULTS_PATH="output/results/eval_results.json"
FIGURES_DIR="output/figures"
CONFIG_PATH="systemx-jupyter/models/config/default_reversed.yaml"
EPOCHS_HGT=30
EPOCHS_MLP=50
FORCE=""
SKIP_TRAINING=false
SKIP_EVALUATION=false
VARIANTS_FLAG=""

_SCRIPT_DIR="SystemX/experiment/ablation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --annotations_dir) ANNOTATIONS_DIR="$2"; shift 2 ;;
    --raw_dir)         RAW_DIR="$2"; shift 2 ;;
    --output_dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --results_path)    RESULTS_PATH="$2"; shift 2 ;;
    --figures_dir)     FIGURES_DIR="$2"; shift 2 ;;
    --epochs_hgt)      EPOCHS_HGT="$2"; shift 2 ;;
    --epochs_mlp)      EPOCHS_MLP="$2"; shift 2 ;;
    --variants)        shift; VARIANTS_FLAG="--variants"; while [[ $# -gt 0 && "$1" != --* ]]; do VARIANTS_FLAG="$VARIANTS_FLAG $1"; shift; done ;;
    --force)           FORCE="--force"; shift ;;
    --skip_training)   SKIP_TRAINING=true; shift ;;
    --skip_evaluation) SKIP_EVALUATION=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

MANIFEST_PATH="${OUTPUT_DIR}/manifest.json"

echo "============================================================"
echo "  SystemX Experiment Pipeline"
echo "============================================================"
echo "  annotations_dir : ${ANNOTATIONS_DIR}"
echo "  raw_dir         : ${RAW_DIR}"
echo "  checkpoints     : ${OUTPUT_DIR}"
echo "  results_path    : ${RESULTS_PATH}"
echo "  figures_dir     : ${FIGURES_DIR}"
echo "  epochs_hgt      : ${EPOCHS_HGT}"
echo "  epochs_mlp      : ${EPOCHS_MLP}"
echo "  force           : ${FORCE:-no}"
echo "  skip_training   : ${SKIP_TRAINING}"
echo "  skip_evaluation : ${SKIP_EVALUATION}"
echo "============================================================"

if [ "${SKIP_TRAINING}" = false ]; then
  echo ""
  echo ">>> STEP 1: Training all model variants"
  "${PYTHON}" "${_SCRIPT_DIR}/train_all.py" \
    --annotations_dir "${ANNOTATIONS_DIR}" \
    --output_dir      "${OUTPUT_DIR}" \
    --epochs_hgt      "${EPOCHS_HGT}" \
    --epochs_mlp      "${EPOCHS_MLP}" \
    ${FORCE}
  echo ">>> Training complete."
else
  echo ">>> STEP 1: Skipped (--skip_training)."
fi

if [ "${SKIP_EVALUATION}" = false ]; then
  echo ""
  echo ">>> STEP 2: Evaluating all ablation variants"
  "${PYTHON}" "${_SCRIPT_DIR}/evaluate_all.py" \
    --raw_dir         "${RAW_DIR}" \
    --annotations_dir "${ANNOTATIONS_DIR}" \
    --config_path     "${CONFIG_PATH}" \
    --manifest_path   "${MANIFEST_PATH}" \
    --output_path     "${RESULTS_PATH}" \
    ${VARIANTS_FLAG} \
    ${FORCE}
  echo ">>> Evaluation complete."
else
  echo ">>> STEP 2: Skipped (--skip_evaluation)."
fi

echo ""
echo ">>> STEP 3: Generating paper figures"
"${PYTHON}" "${_SCRIPT_DIR}/plot_results.py" \
  --results_path "${RESULTS_PATH}" \
  --output_dir   "${FIGURES_DIR}"
echo ">>> Figures saved to ${FIGURES_DIR}/"

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Checkpoints : ${MANIFEST_PATH}"
echo "  Results     : ${RESULTS_PATH}"
echo "  Figures     : ${FIGURES_DIR}/"
echo "============================================================"
