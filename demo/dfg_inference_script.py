import logging
import yaml
from pathlib import Path

import nbformat
import torch
import os

from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.notebook import Notebook
from ApexDAG.nn.training import GraphProcessor, GraphEncoder
from ApexDAG.experiments.pretrain import create_model 
from ApexDAG.sca.constants import REVERSE_DOMAIN_EDGE_TYPES
    
if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description="Finetune GAT model")
    parser.add_argument("--notebook_path", type=str, default="demo/temp/notebooks/demo_blank.ipynb", help="Path to the notebook of interest!")
    parser.add_argument("--checkpoint_path", type=str, default="demo/temp/graphs", help="Path to the dfg files!")
    parser.add_argument("--encoded_checkpoint_path", type=str, default="data/raw/pytorch-embedded-inference-reversed", help="Path of the encoded graph files")
    
    # read noetebook object from notebook_path
    args = parser.parse_args()
    
    # os.makedirs(args.checkpoint_path, exist_ok=True)
    
    # notebook_path = Path(args.notebook_path)
    # with open(notebook_path, "r", encoding="utf-8") as f:
    #     notebook_object = nbformat.read(f, as_version=4)
    # notebook = Notebook(url=None, nb=notebook_object)
    # notebook.create_execution_graph(greedy=True)
    
    # dfg = DataFlowGraph()
    # dfg.parse_notebook(notebook)
    # dfg.optimize()
    # dfg.save_dfg(os.path.join(args.checkpoint_path, f"temp.execution_graph"))
    
    config = yaml.safe_load(open("ApexDAG/experiments/configs/finetune/default_reversed.yaml", "r"))
    
    checkpoint_path = Path(args.checkpoint_path)
    encoded_checkpoint_path = Path(args.encoded_checkpoint_path).parent / "pytorch-encoded-inference"
    logger = logging.getLogger("inference_finetuned_gat")
    
    graph_processor = GraphProcessor(checkpoint_path, logger)
    graph_encoder = GraphEncoder(encoded_checkpoint_path, 
                                 logger, 
                                 min_nodes = 3, 
                                 min_edges = 2, 
                                 load_encoded_old_if_exist = True,
                                 mode = 'REVERSED',
    )
    
    graph_processor.load_preprocessed_graphs()
    encoded_graphs = graph_encoder.encode_graphs(
            graph_processor.graphs, feature_to_encode="domain_label"
        )
    model = create_model(
        config=config,
        reversed=True,
        tasks=["node_classification"]
    )
    checkpoint = torch.load("demo/checkpoints/model_epoch_finetuned_GraphTransformsMode.REVERSED_440.pt",
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')
    model.eval()

    results = []
    with torch.no_grad():
        for i, (graph_encoded, graph) in enumerate(zip(encoded_graphs, graph_processor.graphs)):
            output = model(graph_encoded)
            labels = torch.argmax(output['node_type_preds'], dim=1)
            labels_names = [REVERSE_DOMAIN_EDGE_TYPES[label.item()] for label in labels]
            results.append(labels)
            # print graph edges 
            for edge, label in zip(graph.edges, labels_names):
                print(f"Edge {edge} [{label}]: {edge[0]} -{graph.edges._adjdict[edge[0]][edge[1]]['code']}-> {edge[1] }")
            print(f"Graph {i}: Output of network! {labels_names}")

    print("Inference complete. Number of graphs processed:", len(results))
    