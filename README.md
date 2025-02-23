# APEX-DAG: <ins>A</ins>utomating <ins>P</ins>ipeline <ins>EX</ins>traction with <ins>D</ins>ataflow, Static Code <ins>A</ins>nalysis, and <ins>G</ins>raph Attention Networks

## Abstract

Modern data-driven systems often rely on complex pipelines to
process and transform data for downstream machine learning (ML)
tasks. Extracting these pipelines and understanding their struc-
ture is critical for ensuring transparency, performance optimiza-
tion, and maintainability, especially in large-scale projects. In this
work, we introduce a novel system, APEX-DAG (Automating Pipeline
EXtraction with Dataflow, Static Code Analysis, and Graph Atten-
tion Networks), which automates the extraction of data pipelines
from computational notebooks or scripts. Unlike execution-based
methods, APEX-DAG leverages static code analysis to identify the
dataflow, transformations, and dependencies within ML workflows
without executing the code or the need to alter the code. Further,
after an initial training phase, our system can identify pipelines
that built with previously unseen libraries.

## Demo video
You can find the demo video <a href="https://drive.google.com/file/d/1Al18W68A5X8hl3LfdGuOFVs1WUtNogjB/view?usp=sharing">here</a>.

## Environment setup with conda:

```
conda create --name apexdag python=3.10
conda activate apexdag
pip install -r requirements.txt
pip install -e .

conda install --channel conda-forge pygraphviz // requires graphviz

```

## Progress

- [x] Demonstration Paper (ACM SIGMOD/PODS 2025, Berlin, Germany)
- [x] Translator (Dataflow Extraction, Tokenizing, Encoding Graph)
- [x] Detector & Estimator (Graph Attention Network, Training Tasks, Dataset)
- [x] Annotator (Graph Attention Network, Training Tasks, Dataset)