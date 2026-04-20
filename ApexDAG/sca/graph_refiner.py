from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph
from ApexDAG.sca import (
    NODE_TYPES,
    EDGE_TYPES,
    REVERSE_DOMAIN_EDGE_TYPES,
    DOMAIN_EDGE_TYPES,
    DOMAIN_NODE_TYPES,
)


class GraphRefiner:
    def refine(self, graph: PythonDataFlowGraph) -> None:
        G = graph.get_graph()
        
        UPGRADABLE_NODE_TYPES = {
            NODE_TYPES["VARIABLE"], 
            NODE_TYPES["INTERMEDIATE"]
        }
        
        node_type_updates = {}
        edge_domain_updates = {}
        edge_numeric_updates = {}
        
        known_datasets = set()
        known_models = set()

        for n, data in G.nodes(data=True):
            if data.get("node_type") == DOMAIN_NODE_TYPES["DATASET"]:
                known_datasets.add(n)
            elif data.get("node_type") == DOMAIN_NODE_TYPES["MODEL"]:
                known_models.add(n)

        for u, v, key, data in G.edges(keys=True, data=True):
            domain_label = data.get("predicted_label", "")
            if domain_label not in REVERSE_DOMAIN_EDGE_TYPES:
                continue
                
            edge_type_name = REVERSE_DOMAIN_EDGE_TYPES[domain_label]
            
            if edge_type_name == "DATA_IMPORT_EXTRACTION":
                known_datasets.add(v)
            elif edge_type_name in ["MODEL_TRAIN", "MODEL_EVALUATION", "HYPERPARAMETER_TUNING"]:
                known_models.add(v)

        changed = True
        while changed:
            changed = False
            for u, v, key, data in G.edges(keys=True, data=True):
                domain_label = data.get("predicted_label", "")
                if domain_label not in REVERSE_DOMAIN_EDGE_TYPES:
                    continue
                    
                edge_type_name = REVERSE_DOMAIN_EDGE_TYPES[domain_label]
                
                if edge_type_name == "DATA_TRANSFORM":
                    if u in known_datasets and v not in known_datasets:
                        known_datasets.add(v)
                        changed = True
                    if u in known_models and v not in known_models:
                        known_models.add(v)
                        changed = True

        for n in known_datasets:
            if G.nodes[n].get("node_type") in UPGRADABLE_NODE_TYPES:
                node_type_updates[n] = DOMAIN_NODE_TYPES["DATASET"]
                
        for n in known_models:
            if G.nodes[n].get("node_type") in UPGRADABLE_NODE_TYPES:
                node_type_updates[n] = DOMAIN_NODE_TYPES["MODEL"]

        for u, v, key, data in G.edges(keys=True, data=True):
            if v in known_datasets and G.nodes[u].get("node_type") == NODE_TYPES["LITERAL"]:
                domain_label = data.get("predicted_label", "")
                
                if domain_label not in REVERSE_DOMAIN_EDGE_TYPES or REVERSE_DOMAIN_EDGE_TYPES[domain_label] != "DATA_IMPORT_EXTRACTION":
                    edge_domain_updates[(u, v, key)] = "DATA_IMPORT_EXTRACTION"
                    edge_numeric_updates[(u, v, key)] = DOMAIN_EDGE_TYPES["DATA_IMPORT_EXTRACTION"]

        for n, data in G.nodes(data=True):
            if data.get("node_type") == NODE_TYPES["LITERAL"]:
                node_type_updates[n] = DOMAIN_NODE_TYPES["LITERAL"]

        if node_type_updates:
            graph.set_domain_node_label(node_type_updates, name="node_type")
            
        if edge_domain_updates:
            graph.set_domain_label(edge_domain_updates, name="domain_label")
            graph.set_domain_label(edge_numeric_updates, name="predicted_label") 
            graph.set_domain_label(edge_numeric_updates, name="edge_type")
