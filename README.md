# Dissertation

![Target picture](docs/system-overview.png)

## WiP

- [ ] Paper Teaser
- [x] Demonstration Paper (ACM SIGMOD/PODS 2025, Berlin, Germany)
- [x] Encoder (Dataflow Extraction, Tokenizing, Encoding Graph)
- [ ] Detector & Estimator (Graph Attention Network, Training Tasks, Dataset)
- [ ] Decoder (Reconstructing, Highlighting, Filter)

![AST Graph](docs/ast_graph.jpg)


# Environment setup with conda:
```
conda create --name my_env python=3.8
conda activate my_env
pip install -r requirements.txt

conda install --channel conda-forge pygraphviz // requires graphviz
```