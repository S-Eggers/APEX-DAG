#!/bin/bash
# KB Proposal Evaluation Runner
# Convenience script for running KB improvement iterations

set -e

# Function to display usage
usage() {
    echo "Usage: $0 {catboost|sklearn} [options]"
    echo ""
    echo "Libraries:"
    echo "  catboost    Evaluate CatBoost corpus"
    echo "  sklearn     Evaluate Scikit-learn corpus"
    echo ""
    echo "Options:"
    echo "  --corpus PATH           Override default corpus path"
    echo "  --output PATH           Override default output directory"
    echo "  --iterations N          Number of iterations (default: 200)"
    echo "  --kb-path PATH          Path to existing KB to continue from"
    echo ""
    echo "Examples:"
    echo "  $0 catboost"
    echo "  $0 sklearn --iterations 50"
    echo "  $0 catboost --output output/catboost_run2 --kb-path output/final_kb.csv"
    exit 1
}

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    usage
fi

LIBRARY=$1
shift

# Default paths - modify these for your system
DEFAULT_CATBOOST_CORPUS="/home/nina/projects/APEXDAG_datasets/catboost"
DEFAULT_SKLEARN_CORPUS="/home/nina/projects/APEXDAG_datasets/sklearn"

# Set library-specific defaults
case $LIBRARY in
    catboost)
        CORPUS_PATH=${CORPUS_PATH:-$DEFAULT_CATBOOST_CORPUS}
        DOCS_URL="https://catboost.ai/docs/en/"
        OUTPUT_DIR="output/proposals/catboost"
        echo "=========================================="
        echo "🚀 CatBoost KB Proposal Evaluation"
        echo "=========================================="
        ;;
    sklearn)
        CORPUS_PATH=${CORPUS_PATH:-$DEFAULT_SKLEARN_CORPUS}
        DOCS_URL="https://scikit-learn.org/stable/api/index.html"
        OUTPUT_DIR="output/proposals/sklearn"
        echo "=========================================="
        echo "🚀 Scikit-learn KB Proposal Evaluation"
        echo "=========================================="
        ;;
    *)
        echo "Error: Unknown library '$LIBRARY'"
        echo ""
        usage
        ;;
esac

# Default values for optional arguments
ITERATIONS=200
KB_PATH=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --corpus)
            CORPUS_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --kb-path)
            KB_PATH="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option '$1'"
            usage
            ;;
    esac
done

# Verify corpus path exists
if [ ! -d "$CORPUS_PATH" ]; then
    echo "Error: Corpus path does not exist: $CORPUS_PATH"
    exit 1
fi

# Display configuration
echo "Corpus:      $CORPUS_PATH"
echo "Documentation: $DOCS_URL"
echo "Output:      $OUTPUT_DIR"
echo "Iterations:  $ITERATIONS"
if [ -n "$KB_PATH" ]; then
    echo "Existing KB: $KB_PATH"
fi
echo "=========================================="
echo ""

# Build command
CMD="python ApexDAG/vamsa/kb_advancement/propose_changes.py \
    --corpus \"$CORPUS_PATH\" \
    --docs \"$DOCS_URL\" \
    --output \"$OUTPUT_DIR\" \
    --iterations $ITERATIONS"

if [ -n "$KB_PATH" ]; then
    CMD="$CMD --kb-path \"$KB_PATH\""
fi

# Run the command
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "✅ Completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
