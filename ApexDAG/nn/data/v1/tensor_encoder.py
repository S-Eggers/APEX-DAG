import logging
import random

import networkx as nx
import torch
from torch_geometric.data import Data

from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES
from ApexDAG.util.logger import configure_apexdag_logger
from ApexDAG.util.training_utils import (
    InsufficientNegativeEdgesException,
    InsufficientPositiveEdgesException,
)

from .embedding import Embedding, EmbeddingType

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class Encoder:
    def __init__(
        self,
        embedding_model: Embedding = None,
        nude_num_types_pretrain: int = 8,
        edge_num_types_pretrain: int = 6,
        min_edges: int = 2,
    ) -> None:
        self._embedding = embedding_model or Embedding(EmbeddingType.FASTTEXT, logger)
        self.nude_num_types_pretrain = nude_num_types_pretrain
        self.edge_num_types_pretrain = edge_num_types_pretrain
        self.min_edges = min_edges

    def encode(self, graph: nx.MultiDiGraph, feature_to_encode: str) -> Data:
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        edges = [(node_to_id[u], node_to_id[v]) for u, v in graph.edges()]

        negative_edges = self._sample_negative_edges(graph, node_to_id, len(edges))

        all_edges = edges + list(negative_edges)
        source_nodes, target_nodes = (
            zip(*all_edges, strict=False) if all_edges else ([], [])
        )

        source_nodes_tensor = torch.tensor(source_nodes, dtype=torch.long)
        target_nodes_tensor = torch.tensor(target_nodes, dtype=torch.long)

        node_features, node_types = self._extract_node_features(graph)
        edge_features, edge_types, edge_existence = self._extract_edge_features(
            graph, negative_edges, feature_to_encode
        )

        return Data(
            x=node_features,
            node_types=torch.tensor(node_types, dtype=torch.long),
            edge_index=torch.stack([source_nodes_tensor, target_nodes_tensor], dim=0)
            if all_edges
            else torch.empty((2, 0), dtype=torch.long),
            edge_types=edge_types,
            edge_features=edge_features,
            edge_existence=edge_existence,
        )

    def encode_reversed(self, graph: nx.MultiDiGraph, feature_to_encode: str) -> Data:
        new_nodes = list(graph.edges())
        node_to_id = {edge: idx for idx, edge in enumerate(new_nodes)}

        new_edges = []
        new_edges_indices = []

        for node_idx, node in enumerate(graph.nodes()):
            connected_edges_out = [(u, v) for (u, v) in graph.edges() if u == node]
            connected_edges_in = [(u, v) for (u, v) in graph.edges() if v == node]

            if len(connected_edges_out) == 1:
                new_edges.append(
                    (
                        node_to_id[connected_edges_out[0]],
                        node_to_id[connected_edges_out[0]],
                    )
                )
                new_edges_indices.append(node_idx)

            if len(connected_edges_in) == 1:
                new_edges.append(
                    (
                        node_to_id[connected_edges_in[0]],
                        node_to_id[connected_edges_in[0]],
                    )
                )
                new_edges_indices.append(node_idx)

            if len(connected_edges_in) == 1 and len(connected_edges_out) == 1:
                new_edges.append(
                    (
                        node_to_id[connected_edges_in[0]],
                        node_to_id[connected_edges_out[0]],
                    )
                )
                new_edges_indices.append(node_idx)

            for i in range(len(connected_edges_out)):
                for j in range(i + 1, len(connected_edges_out)):
                    new_edges.append(
                        (
                            node_to_id[connected_edges_out[i]],
                            node_to_id[connected_edges_out[j]],
                        )
                    )
                    new_edges_indices.append(node_idx)

            for i in range(len(connected_edges_in)):
                for j in range(i + 1, len(connected_edges_in)):
                    new_edges.append(
                        (
                            node_to_id[connected_edges_in[i]],
                            node_to_id[connected_edges_in[j]],
                        )
                    )
                    new_edges_indices.append(node_idx)

            for in_edge in connected_edges_in:
                for out_edge in connected_edges_out:
                    new_edges.append((node_to_id[in_edge], node_to_id[out_edge]))
                    new_edges_indices.append(node_idx)

        source_nodes, target_nodes = (
            zip(*new_edges, strict=False) if new_edges else ([], [])
        )
        source_nodes_tensor = torch.tensor(source_nodes, dtype=torch.long)
        target_nodes_tensor = torch.tensor(target_nodes, dtype=torch.long)

        node_features, node_types, _ = self._extract_edge_features(
            graph, [], feature_to_encode
        )
        edge_features, edge_types = self._extract_node_features(graph)

        edge_features_tensor = (
            torch.stack([edge_features[i] for i in new_edges_indices])
            if edge_features is not None
            else None
        )
        edge_types_tensor = (
            torch.tensor([edge_types[i] for i in new_edges_indices], dtype=torch.long)
            if edge_types is not None
            else None
        )

        return Data(
            x=node_features,
            node_types=node_types,
            edge_index=torch.stack([source_nodes_tensor, target_nodes_tensor], dim=0)
            if new_edges
            else torch.empty((2, 0), dtype=torch.long),
            edge_types=edge_types_tensor,
            edge_features=edge_features_tensor,
        )

    def _sample_negative_edges(self, graph, node_to_id, num_neg_samples):
        nodes = list(graph.nodes())
        negative_edges = set()
        existing_edges = set((u, v) for u, v in graph.edges()) | set(
            (v, u) for u, v in graph.edges()
        )

        max_possible_edges = len(nodes) * (len(nodes) - 1)
        if len(existing_edges) >= max_possible_edges:
            raise InsufficientNegativeEdgesException()

        if max_possible_edges < len(existing_edges) + num_neg_samples:
            logger.warning(
                f"Not enough negative edges. Found {len(existing_edges)}, max {max_possible_edges}."
            )
            num_neg_samples = max_possible_edges - len(existing_edges)

        while len(negative_edges) < num_neg_samples:
            u, v = random.sample(nodes, 2)
            if (u, v) not in graph.edges and (v, u) not in graph.edges:
                negative_edges.add((node_to_id[u], node_to_id[v]))

        return negative_edges

    def _extract_node_features(self, graph):
        node_features, node_types = [], []

        for _, attrs in graph.nodes(data=True):
            raw_code = attrs.get("code", "")
            label = attrs.get("label", "")

            semantic_text = (
                str(raw_code) if raw_code and raw_code != "None" else str(label)
            )

            if "\n" in semantic_text:
                semantic_text = semantic_text.replace("\n", " ")

            embedding = self._embedding.embed(semantic_text)
            node_type = int(attrs.get("node_type", -1))

            node_features.append(embedding)
            node_types.append(node_type)

        return (torch.stack(node_features), node_types)

    def _extract_edge_features(
        self, graph, negative_edges, feature_to_encode="edge_type"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_features, edge_types, edge_existence = [], [], []

        def process_edge(attrs, is_positive) -> None:
            def map_labels_to_int(label):
                if isinstance(label, str):
                    return DOMAIN_EDGE_TYPES.get(label, -1)
                if isinstance(label, int):
                    return label
                return -1

            edge_emb = (
                self._embedding.embed(attrs.get("label", ""))
                if is_positive
                else torch.zeros(self._embedding.dimension)
            )

            edge_type_for_training = (
                int(map_labels_to_int(attrs.get(feature_to_encode, -1)))
                if is_positive
                else -1
            )

            edge_features.append(edge_emb)
            edge_types.append(edge_type_for_training)
            edge_existence.append(1.0 if is_positive else 0.0)

        for _, _, attrs in graph.edges(data=True):
            process_edge(attrs, is_positive=True)

        for _u, _v in negative_edges:
            process_edge({}, is_positive=False)

        if len(edge_features) < self.min_edges:
            raise InsufficientPositiveEdgesException(
                f"Graph has insufficient positive edges: {len(edge_features)} < {self.min_edges}"
            )

        return (
            torch.stack(edge_features),
            torch.tensor(edge_types, dtype=torch.long),
            torch.tensor(edge_existence, dtype=torch.float32),
        )
