import json
import torch
from torch_geometric.data import Data

def load_cytoscape_json_to_pyg(json_path: str) -> Data:
    with open(json_path, 'r') as f:
        elements = json.load(f)
        
    node_mapping = {} 
    node_features = []
    node_labels = []
    
    edge_sources = []
    edge_targets = []
    edge_labels = []

    for idx, el in enumerate([e for e in elements if e["group"] == "nodes"]):
        data = el["data"]
        node_mapping[data["id"]] = idx
        
        node_features.append([data.get("node_type", 0)]) 
        node_labels.append(data.get("predicted_label", 0)) 

    for el in [e for e in elements if e["group"] == "edges"]:
        data = el["data"]
        src = node_mapping.get(data["source"])
        tgt = node_mapping.get(data["target"])
        
        if src is not None and tgt is not None:
            edge_sources.append(src)
            edge_targets.append(tgt)
            edge_labels.append(data.get("predicted_label", 0))

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)
    edge_attr = torch.tensor(edge_labels, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)