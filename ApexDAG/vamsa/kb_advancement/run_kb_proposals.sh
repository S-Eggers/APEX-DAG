#!/bin/bash
# KB Proposal Evaluation - Run Commands

# CatBoost
python ApexDAG/vamsa/kb_advancement/propose_changes.py \
    --corpus /home/nina/projects/APEXDAG_datasets/catboost \
    --docs https://catboost.ai/docs/en/ \
    --output output/proposals/catboost \
    --iterations 200

# Scikit-learn
# python ApexDAG/vamsa/kb_advancement/propose_changes.py \
#     --corpus /home/nina/projects/APEXDAG_datasets/sklearn \
#     --docs https://scikit-learn.org/stable/api/index.html \
#     --output output/proposals/sklearn \
#     --iterations 200
