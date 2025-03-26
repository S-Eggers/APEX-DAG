import random
import fasttext
import networkx as nx
import torch
from torch_geometric.data import Data
from ApexDAG.util.training_utils import InsufficientNegativeEdgesException
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES

class Encoder:
    def __init__(self, logger=None, nude_num_types_pretrain = 8, edge_num_types_pretrain = 6):
        self.logger = logger or print
        self._fasttext_model = fasttext.load_model("cc.en.300.bin")
        self.nude_num_types_pretrain = nude_num_types_pretrain
        self.edge_num_types_pretrain = edge_num_types_pretrain
        
       
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

        return (
            torch.stack(edge_features),
            torch.tensor(edge_types, dtype=torch.long),
            torch.tensor(edge_existence, dtype=torch.float32),
        )

def swap_nodes_and_edges_in_data(data: Data) -> Data:
    edge_id_to_node = {i: (int(u),int(v)) for i, (u, v) in enumerate(zip(data.edge_index[0], data.edge_index[1]))}
    
    node_pair_to_edge_id = {(int(u),int(v)): i for i, (u, v) in edge_id_to_node.items()}
    node_pair_to_edge_id.update({(int(u),int(v)): i for i, (u, v) in edge_id_to_node.items()})  # Add reverse pairs

    new_edge_index = []
    new_edge_features = []  # store the new edges' features 
    new_edge_types = []  #  store the new edges' types  
    for edge_id_1, (u, v) in edge_id_to_node.items():
        u = int(u)
        v = int(v)
        
        neighbors_u = data.edge_index[1][data.edge_index[0] == u].tolist()
        neighbors_v = data.edge_index[1][data.edge_index[0] == v].tolist()

        for neighbor_u in neighbors_u:
            if neighbor_u != v:  # Avoid self-loops
                edge_id_2 = node_pair_to_edge_id.get((neighbor_u, u)) 
                if edge_id_2 is not None:
                    new_edge_index.append([edge_id_2, edge_id_1])
                    new_edge_features.append(data.x[u])  # Use node u's features for the new edge
                    new_edge_types.append(data.node_types[u]) 

        for neighbor_v in neighbors_v:
            if neighbor_v != u:  # Avoid self-loops
                edge_id_2 = node_pair_to_edge_id.get((v, neighbor_v)) 
                if edge_id_2 is not None:
                    new_edge_index.append([edge_id_1, edge_id_2])
                    new_edge_features.append(data.x[v])  # v's features for the new edge
                    new_edge_types.append(data.node_types[v])

    new_edge_index = torch.tensor(new_edge_index).t().contiguous()
    new_edge_features = torch.stack(new_edge_features) 
    new_edge_types = torch.tensor(new_edge_types)

    return Data(
        x=data.edge_features,                # Original node features are now edge features
        node_types=data.edge_types,          # Edge types are now the original node types
        edge_index=new_edge_index,          # New edge index defines the new connectivity
        edge_types=new_edge_types,          # Edge types are now the original edge types
        edge_features=new_edge_features,    # Edge features are now the original node features
    )
