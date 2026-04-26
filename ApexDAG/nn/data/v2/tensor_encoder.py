import logging
import random

import networkx as nx
import torch
from torch_geometric.data import Data

from ApexDAG.nn.data.v2.embedding import CodeBERTEmbedding
from ApexDAG.nn.data.v2.pruner import GraphPruner
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES

logger = logging.getLogger(__name__)


class InsufficientPositiveEdgesException(Exception):
    pass


class EncoderV2:
    def __init__(
        self,
        embedding_model: CodeBERTEmbedding,
        pruner: GraphPruner,
        min_edges: int = 2,
        negative_sample_ratio: float = 1.0,
    ) -> None:
        self._embedding = embedding_model
        self._pruner = pruner
        self.min_edges = min_edges
        self.negative_sample_ratio = negative_sample_ratio

    def encode(self, raw_graph: nx.MultiDiGraph) -> Data:
        graph = self._pruner.prune(raw_graph)

        nodes = list(graph.nodes(data=True))
        edges = list(graph.edges(data=True))

        if len(edges) < self.min_edges:
            raise InsufficientPositiveEdgesException(
                f"Pruned graph has insufficient edges: {len(edges)} < {self.min_edges}"
            )

        node_to_idx = {node_id: idx for idx, (node_id, _) in enumerate(nodes)}

        x_features = []
        node_types = []

        for _, attrs in nodes:
            raw_code = attrs.get("code", "")
            label = attrs.get("label", "")
            semantic_text = (
                str(raw_code) if raw_code and raw_code != "None" else str(label)
            )
            x_features.append(self._embedding.embed(semantic_text))
            node_types.append(int(attrs.get("node_type", -1)))

        X = torch.stack(x_features)
        Y_node = torch.tensor(node_types, dtype=torch.long)

        edge_sources = []
        edge_targets = []
        edge_features = []
        edge_labels = []
        edge_existence = []

        for u, v, attrs in edges:
            edge_sources.append(node_to_idx[u])
            edge_targets.append(node_to_idx[v])

            operation_text = str(attrs.get("label", ""))
            edge_features.append(self._embedding.embed(operation_text))

            domain_lbl = attrs.get("predicted_label", attrs.get("edge_type", -1))
            edge_labels.append(int(domain_lbl))
            edge_existence.append(1.0)

        num_neg_samples = int(len(edges) * self.negative_sample_ratio)
        neg_edges = self._sample_negative_edges(graph, node_to_idx, num_neg_samples)

        for u_idx, v_idx in neg_edges:
            edge_sources.append(u_idx)
            edge_targets.append(v_idx)
            edge_features.append(
                torch.zeros(self._embedding.dimension, dtype=torch.float32)
            )
            edge_labels.append(DOMAIN_EDGE_TYPES["NOT_RELEVANT"])
            edge_existence.append(0.0)

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.stack(edge_features)  # Shape: [num_edges, 768]
        Y_edge = torch.tensor(edge_labels, dtype=torch.long)
        Y_exist = torch.tensor(edge_existence, dtype=torch.float32)

        return Data(
            x=X,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_node=Y_node,
            y_edge=Y_edge,
            y_exist=Y_exist,
        )

    def _sample_negative_edges(self, graph, node_to_idx, num_neg_samples) -> set:
        nodes = list(graph.nodes())
        negative_edges = set()

        existing_edges = set()
        for u, v in graph.edges():
            existing_edges.add((u, v))
            existing_edges.add((v, u))

        max_possible = len(nodes) * (len(nodes) - 1)
        if len(existing_edges) >= max_possible:
            return set()

        actual_samples = min(num_neg_samples, max_possible - len(existing_edges))

        while len(negative_edges) < actual_samples:
            u, v = random.sample(nodes, 2)
            if (u, v) not in existing_edges:
                negative_edges.add((node_to_idx[u], node_to_idx[v]))
                existing_edges.add((u, v))

        return negative_edges
