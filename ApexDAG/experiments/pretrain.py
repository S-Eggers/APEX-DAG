import os
import tqdm
import torch
import logging
from torch.utils.data import random_split

from ApexDAG.encoder import Encoder
from ApexDAG.notebook import Notebook
from ApexDAG.util.kaggle_dataset_iterator import KaggleDatasetIterator

from ApexDAG.sca.constants import EDGE_TYPES, NODE_TYPES
from ApexDAG.sca.graph_utils import save_graph, load_graph
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph

from ApexDAG.nn.gat import MultiTaskGAT
from ApexDAG.nn.dataset import GraphDataset
from ApexDAG.nn.trainer import PretrainingTrainer


def check_graph(G):
    for node, data in G.nodes(data=True):
        for key in data:
            if data[key] is None:
                print(node, key, data[key])
                data[key] = "None" 
            else:
                data[key] = str(data[key])

    for u, v, data in G.edges(data=True):
        for key in data:
            print(u, v, key, data[key])
            if data[key] is None:
                data[key] = "None"
            else:
                data[key] = str(data[key])

def pretrain_gat(args, logger: logging.Logger) -> None:
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = os.path.join(os.getcwd(), "data", "raw", "pretrain-graphs")

    logger.info("Checkpoint path: %s", checkpoint_path)
    if os.path.exists(checkpoint_path):
        logger.info("Loading preprocessed graphs")
        graphs = [load_graph(os.path.join(checkpoint_path, graph)) for graph in tqdm.tqdm(os.listdir(checkpoint_path), desc="Loading graphs")]
    else:
        kaggle_iterator = KaggleDatasetIterator(os.path.join(args.notebook))

        logger.info("Mining dataflows on Kaggle dataset")
        graphs = []
        for competition in kaggle_iterator:
            for notebook_file in competition["ipynb_files"]:
                notebook_path = os.path.join(competition["subfolder_path"], notebook_file)
                notebook = Notebook(notebook_path)
                notebook.create_execution_graph(greedy=args.greedy)
                dfg = DataFlowGraph(notebook_file)
                dfg.parse_notebook(notebook)
                dfg.optimize()
                graphs.append(dfg.get_graph())

        os.makedirs(checkpoint_path, exist_ok=True)
        for index, graph in enumerate(graphs):
            check_graph(graph)
            save_graph(graph, os.path.join(checkpoint_path, f"graph_{index}.gml"))

    checkpoint_path += "-encoded"
    if os.path.exists(checkpoint_path):
        logger.info("Loading encoded graphs")
        encoded_graphs = [
            torch.load(os.path.join(checkpoint_path, path))
            for path
            in tqdm.tqdm(os.listdir(checkpoint_path), desc="Loading encoded graphs")
        ]
    else:
        logger.info("Encoding graphs")
        encoder = Encoder()
        encoded_graphs = [
            encoder.encode(graph)
            for graph
            in tqdm.tqdm(graphs, desc="Encoding graphs")
        ]
        os.makedirs(checkpoint_path, exist_ok=True)
        for index, graph in enumerate(encoded_graphs):
            torch.save(graph, os.path.join(checkpoint_path, f"graph_{index}.pt"))

    logger.info("Creating dataset")
    dataset = GraphDataset(encoded_graphs)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = MultiTaskGAT(hidden_dim=300, num_heads=4, node_classes=len(NODE_TYPES), edge_classes=len(EDGE_TYPES))
    print(model)
    # Instantiate the trainer
    trainer = PretrainingTrainer(model, train_dataset, val_dataset, device="cpu")
    # Train the model
    trainer.train(num_epochs=100)
