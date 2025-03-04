import random
import fasttext
import networkx as nx
import torch
from torch_geometric.data import Data
from ApexDAG.util.training_utils import InsufficientNegativeEdgesException

class Encoder:
    def __init__(self, logger=None):
        self.logger = logger or print
        self._fasttext_model = fasttext.load_model("cc.en.300.bin")
        self._node_type_embedding = torch.nn.Embedding(300, 50)
        self._edge_type_embedding = torch.nn.Embedding(300, 100)

    def encode(self, graph: nx.MultiDiGraph) -> Data:
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        edges = [(node_to_id[u], node_to_id[v]) for u, v in graph.edges()]
        negative_edges = self._sample_negative_edges(graph, node_to_id, len(edges))

        all_edges = edges + list(negative_edges)
        source_nodes, target_nodes = zip(*all_edges) if all_edges else ([], [])
        source_nodes = torch.tensor(source_nodes, dtype=torch.long)
        target_nodes = torch.tensor(target_nodes, dtype=torch.long)

        node_features, node_types, node_combined_features = self._extract_node_features(graph)
        edge_features, edge_types, edge_combined_features, edge_existence = self._extract_edge_features(graph, edges, negative_edges)

        return Data(
            x=node_features,
            node_types=torch.tensor(node_types, dtype=torch.long),
            node_combined_features=node_combined_features,
            edge_index=torch.stack([source_nodes, target_nodes], dim=0) if all_edges else torch.empty((2, 0), dtype=torch.long),
            edge_combined_features=edge_combined_features,
            edge_types=edge_types,
            edge_features=edge_features,
            edge_existence=edge_existence,
        )

    def _get_sentence_vector(self, sentence: str) -> torch.Tensor:
        if "\n" in sentence:
            self.logger(f"ERROR: Sentence contains newline character: {sentence}")
            sentence = sentence.replace("\n", " ")
        # Convert to PyTorch tensor
        return torch.tensor(self._fasttext_model.get_sentence_vector(sentence), dtype=torch.float32)

    def _sample_negative_edges(self, graph, node_to_id, num_neg_samples):
        nodes = list(graph.nodes())
        negative_edges = set()
        existing_edges = set((u, v) for u, v in graph.edges()) | set((v, u) for u, v in graph.edges())

        # check if there are enough negative edges to sample
        if len(existing_edges) >= len(nodes) * (len(nodes) - 1):
            raise InsufficientNegativeEdgesException()
        
        elif len(nodes) * (len(nodes) - 1) < len(existing_edges) + num_neg_samples:
            self.logger(f"WARNING: Not enough negative edges to sample. Found {len(existing_edges)} existing edges, need {len(nodes) * (len(nodes) - 1)}")
            num_neg_samples = len(nodes) * (len(nodes) - 1) - len(existing_edges)

        while len(negative_edges) < num_neg_samples:
            u, v = random.sample(nodes, 2)
            if (u, v) not in graph.edges and (v, u) not in graph.edges:
                negative_edges.add((node_to_id[u], node_to_id[v]))
        return negative_edges

    def _extract_node_features(self, graph):
        node_combined_features, node_features, node_types = [], [], []
        for _, attrs in graph.nodes(data=True):
            variable_name = " ".join(attrs["label"].split("_")[:-1])
            embedding = self._get_sentence_vector(variable_name)
            node_type = int(attrs.get("node_type", 0))
            node_type_emb = self._node_type_embedding(torch.tensor([node_type]))

            combined_emb = torch.cat([embedding, node_type_emb.squeeze(0)])
            node_combined_features.append(combined_emb)
            node_features.append(embedding)
            node_types.append(node_type)

        return (
            torch.stack(node_features) if node_features else torch.empty((0, 300), dtype=torch.float32),
            node_types,
            torch.stack(node_combined_features) if node_combined_features else torch.empty((0, 350), dtype=torch.float32),
        )

    def _extract_edge_features(self, graph, edges, negative_edges) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_combined_features, edge_features, edge_types, edge_existence = [], [], [], []

        def process_edge(attrs, is_positive):
            edge_emb = self._get_sentence_vector(attrs["code"]) if is_positive else torch.zeros(300)
            edge_type = int(attrs.get("edge_type", 0)) if is_positive else 0
            edge_type_emb = self._edge_type_embedding(torch.tensor([edge_type]))
            combined_edge_emb = torch.cat([edge_emb, edge_type_emb.squeeze(0)])

            edge_combined_features.append(combined_edge_emb)
            edge_features.append(edge_emb)
            edge_types.append(edge_type)
            edge_existence.append(1 if is_positive else 0)

        for _, _, attrs in graph.edges(data=True):
            process_edge(attrs, is_positive=True)

        for u, v in negative_edges:
            process_edge({}, is_positive=False)

        return (
            torch.stack(edge_features) if edge_features else torch.empty((0, 300), dtype=torch.float32),
            torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.empty((0,), dtype=torch.long),
            torch.stack(edge_combined_features) if edge_combined_features else torch.empty((0, 400), dtype=torch.float32),
            torch.tensor(edge_existence, dtype=torch.float32) if edge_existence else torch.empty((0,), dtype=torch.float32),
        )

