# APEX-DAG: <ins>A</ins>utomating <ins>P</ins>ipeline <ins>EX</ins>traction with <ins>D</ins>ataflow, Static Code <ins>A</ins>nalysis, and <ins>G</ins>raph Attention Networks
![translator_input_output(1)](https://github.com/user-attachments/assets/2f128811-38ea-4566-acc1-e6d8fd6f1005)

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
You can find the demo video <a href="https://drive.google.com/file/d/18GxG-fEuiqBnU4402_sEy2W1poDhLxGP/view?usp=sharing">here</a>.

## Environment setup with conda:

```
conda create --name apexdag python=3.10
conda activate apexdag
pip install -r requirements.txt
pip install -e .

conda install --channel conda-forge pygraphviz // requires graphviz

```

## Progress

- [x] Translator (Dataflow Extraction, Tokenizing, Encoding Graph)
- [x] Detector & Estimator (Graph Attention Network, Training Tasks, Dataset)
- [x] Annotator (Graph Attention Network, Training Tasks, Dataset)
