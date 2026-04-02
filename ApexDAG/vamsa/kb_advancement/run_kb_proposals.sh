#!/bin/bash
# KB Proposal Evaluation - Run Commands

# Set PYTHONPATH to include project root
export PYTHONPATH="$(pwd):$PYTHONPATH"

# CatBoost
python ApexDAG/vamsa/kb_advancement/propose_changes.py \
    --corpus /home/nina/projects/APEXDAG_datasets/catboost \
    --docs https://catboost.ai/docs/en/ \
    --output output/proposals/catboost \
    --iterations 20 \
    --kb-path output/proposals/annotation_kb_initial.csv

# Scikit-learn
python ApexDAG/vamsa/kb_advancement/propose_changes.py \
    --corpus /home/nina/projects/APEXDAG_datasets/sklearn \
    --docs https://scikit-learn.org/stable/api/index.html \
    --output output/proposals/sklearn \
    --iterations 200 \
    --kb-path output/proposals/catboost/final_annotation_kb.csv
