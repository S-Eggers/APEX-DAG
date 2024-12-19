# Dissertation

![Target picture](docs/system-overview.png)

## WiP

- [x] Dataflow Extraction (from Python ASTs)
- [ ] Code Embedding
- [ ] Data Lineage Detection
- [ ] LLM Finetuning Strategy

![AST Graph](docs/ast_graph.jpg)


# Environment setup with conda:
```
conda create --name my_env python=3.8
conda activate my_env
pip install -r requirements.txt

conda install --channel conda-forge pygraphviz // requires graphviz
```