import random
import fasttext
import networkx as nx
import torch
from torch_geometric.data import Data
from ApexDAG.util.training_utils import InsufficientNegativeEdgesException, InsufficientPositiveEdgesException
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES

class Encoder:
    def __init__(self, logger=None, nude_num_types_pretrain = 8, edge_num_types_pretrain = 6, min_edges = 2):
        self.logger = logger or print
        self._fasttext_model = fasttext.load_model("cc.en.300.bin")
        self.nude_num_types_pretrain = nude_num_types_pretrain
        self.edge_num_types_pretrain = edge_num_types_pretrain
        self.min_edges = min_edges
        
       
    def encode(self, graph: nx.MultiDiGraph, feature_to_encode: str) -> Data:
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        edges = [(node_to_id[u], node_to_id[v]) for u, v in graph.edges()]
        
        negative_edges = self._sample_negative_edges(graph, node_to_id, len(edges))

        all_edges = edges + list(negative_edges)
        source_nodes, target_nodes = zip(*all_edges) if all_edges else ([], [])
        source_nodes = torch.tensor(source_nodes, dtype=torch.long)
        target_nodes = torch.tensor(target_nodes, dtype=torch.long)

        node_features, node_types = self._extract_node_features(graph)
        edge_features, edge_types, edge_existence = self._extract_edge_features(graph, negative_edges, feature_to_encode)

        return Data(
            x=node_features,
            node_types=torch.tensor(node_types, dtype=torch.long),
            edge_index=torch.stack([source_nodes, target_nodes], dim=0) if all_edges else torch.empty((2, 0), dtype=torch.long),
            edge_types=edge_types,
            edge_features=edge_features,
            edge_existence=edge_existence,
        )
    
    def encode_reversed(self, graph: nx.MultiDiGraph, feature_to_encode: str) -> Data:
        # Original edges become nodes
        new_nodes = list(graph.edges())  # Each edge is now a node
        node_to_id = {edge: idx for idx, edge in enumerate(new_nodes)}
        
        new_edges = []
        new_edges_indices = []
        for node_idx, node in enumerate(graph.nodes()):
            # outgoing edges from this node
            connected_edges_out = [(u, v) for (u, v) in graph.edges() if u == node]
            # incoming edges to this node
            connected_edges_in = [(u, v) for  (u, v) in graph.edges() if v == node]
            
            if len(connected_edges_out) == 1:  # Only one outgoing edge
                new_edges.append((node_to_id[connected_edges_out[0]], node_to_id[connected_edges_out[0]]))
                new_edges_indices.append(node_idx)
            
            if len(connected_edges_in) == 1:  # Only one incoming edge
                new_edges.append((node_to_id[connected_edges_in[0]], node_to_id[connected_edges_in[0]]))
                new_edges_indices.append(node_idx)
            
            if len(connected_edges_in) == 1 and len(connected_edges_out) == 1:  # One incoming and one outgoing edge
                new_edges.append((node_to_id[connected_edges_in[0]], node_to_id[connected_edges_out[0]]))
                new_edges_indices.append(node_idx)
            
            # connect outgoing edges to each other
            for i in range(len(connected_edges_out)):
                for j in range(i + 1, len(connected_edges_out)):
                    new_edges.append((node_to_id[connected_edges_out[i]], node_to_id[connected_edges_out[j]]))
                    new_edges_indices.append(node_idx)
            
            #  connect incoming edges to each other
            for i in range(len(connected_edges_in)):
                for j in range(i + 1, len(connected_edges_in)):
                    new_edges.append((node_to_id[connected_edges_in[i]], node_to_id[connected_edges_in[j]]))
                    new_edges_indices.append(node_idx)
            
            # incoming edges to outgoing edges (to simulate node duplication)
            for in_edge in connected_edges_in:
                for out_edge in connected_edges_out:
                    new_edges.append((node_to_id[in_edge], node_to_id[out_edge]))
                    new_edges_indices.append(node_idx)
        
        source_nodes, target_nodes = zip(*new_edges) if new_edges else ([], [])
        source_nodes = torch.tensor(source_nodes, dtype=torch.long)
        target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        
        node_features, node_types, _ = self._extract_edge_features(graph, [], feature_to_encode)  # Extract for new "nodes"
        edge_features, edge_types = self._extract_node_features(graph)  # Extract for new "edges"
        
        edge_features = torch.stack([edge_features[i] for i in new_edges_indices]) if edge_features is not None else None
        edge_types = torch.tensor([edge_types[i] for i in new_edges_indices], dtype=torch.long) if edge_types is not None else None
        
        
        return Data(
            x=node_features,  # Features correspond to edges (which are now nodes)
            node_types=node_types,
            edge_index=torch.stack([source_nodes, target_nodes], dim=0) if new_edges else torch.empty((2, 0), dtype=torch.long),
            edge_types=edge_types,
            edge_features=edge_features,
        )

    def _get_sentence_vector(self, sentence: str) -> torch.Tensor:
        if "\n" in sentence:
            self.logger(f"ERROR: Sentence contains newline character: {sentence}")
            sentence = sentence.replace("\n", " ")
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
        node_features, node_types = [], []
        for _, attrs in graph.nodes(data=True):
            variable_name = " ".join(attrs["label"].split("_")[:-1])
            embedding = self._get_sentence_vector(variable_name)
            node_type = int(attrs.get("node_type", -1))

            node_features.append(embedding)
            node_types.append(node_type)

        return (
            torch.stack(node_features),
            node_types
        )

    def _extract_edge_features(self, graph, negative_edges, feature_to_encode = "edge_type") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_features, edge_types, edge_existence = [], [], []

        def process_edge(attrs, is_positive):
            def map_labels_to_int(label):
                if type(label) == str: # finetune
                    return DOMAIN_EDGE_TYPES[label] if label in DOMAIN_EDGE_TYPES else -1
                if type(label) == int: # pretrain
                    return label
            edge_emb = self._get_sentence_vector(attrs["code"]) if is_positive else torch.zeros(300)
            edge_type_for_training = int(map_labels_to_int(attrs.get(feature_to_encode, -1))) if is_positive else -1 # can be different label

            edge_features.append(edge_emb)
            edge_types.append(edge_type_for_training)
            edge_existence.append(1 if is_positive else 0)

        for _, _, attrs in graph.edges(data=True):
            process_edge(attrs, is_positive=True)

        for u, v in negative_edges:
            process_edge({}, is_positive=False)
            
        if len(edge_features) < self.min_edges:
            raise InsufficientPositiveEdgesException(f"Graph has insufficient positive edges: {len(edge_features)} < {self.min_edges}")

        return (
            torch.stack(edge_features),
            torch.tensor(edge_types, dtype=torch.long),
            torch.tensor(edge_existence, dtype=torch.float32),
        )

