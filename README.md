# APEX-DAG: <ins>A</ins>utomating <ins>P</ins>ipeline <ins>EX</ins>traction with <ins>D</ins>ataflow, Static Code <ins>A</ins>nalysis, and <ins>G</ins>raph Attention Networks

![Target picture](docs/system-overview.png)

## Abstract

Modern data-driven systems often rely on complex pipelines to
process and transform data for downstream machine learning (ML)
tasks. Extracting these pipelines and understanding their structure
is critical for ensuring transparency, performance optimization,
and maintainability, especially in large-scale projects. In this work,
we introduce a novel system, APEX-DAG (Automating Pipeline
EXtraction with Dataflow, Static Code Analysis, and Graph Atten-
tion Networks), which automates the extraction of data pipelines
from computational notebooks or scripts. Unlike execution-based
methods, APEX-DAG leverages static code analysis to identify the
dataflow, transformations, and dependencies within ML workflows
without executing the code or the need to alter the code. Further,
after an initial training phase, our system does not need information
about different libraries in the form of annotation or knowledge
bases. This information is inferred by analyzing the structure and
relationships within extracted dataflow graphs of computational
notebooks and scripts

## Environment setup with conda:

```
conda create --name my_env python=3.8
conda activate my_env
pip install -r requirements.txt

conda install --channel conda-forge pygraphviz // requires graphviz
```

## Progress

- [ ] Paper Teaser
- [x] Demonstration Paper (ACM SIGMOD/PODS 2025, Berlin, Germany)
- [x] Encoder (Dataflow Extraction, Tokenizing, Encoding Graph)
- [ ] Detector & Estimator (Graph Attention Network, Training Tasks, Dataset)
- [ ] Decoder (Reconstructing, Highlighting, Filter)

## Other 

![AST Graph](docs/ast_graph.jpg)