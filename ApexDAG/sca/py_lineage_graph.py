import json
import torch
import networkx as nx
from logging import Logger
from typing import Optional

from ApexDAG.util.logging import setup_logging
from ApexDAG.sca import (
    NODE_TYPES,
    EDGE_TYPES,
    VERBOSE,
    DOMAIN_EDGE_TYPES,
    DOMAIN_NODE_TYPES,
    REVERSE_DOMAIN_EDGE_TYPES,
)
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph
from ApexDAG.label_notebooks.online_labeler import OnlineGraphLabeler as GraphLabeler
from ApexDAG.label_notebooks.utils import Config


class PythonLineageGraph(PythonDataFlowGraph):
    def __init__(
            self, 
            model: dict,  
            notebook_path: str = "", 
            use_llm_backend: bool = False, 
            highlight_relevant: bool = True, 
            replace_dataflow: bool = False
    ):
        super().__init__(notebook_path, replace_dataflow)
        self._logger: Logger = setup_logging(
            f"py_lineage_graph {notebook_path}", VERBOSE
        )
        self.model = model
        self.use_llm_backend = use_llm_backend
        self.highlight_relevant = highlight_relevant

    def parse_code(self, code: str):
        super().parse_code(code)

        self.optimize()
        if self.use_llm_backend:
            self._label_with_llm()
        else:
            self._label_with_gat()

        if self.highlight_relevant:
            self.filter_relevant()

        self._postprocess()
        self.optimize()

    def filter_relevant(self) -> None:
        self._current_state.filter_relevant(lineage_mode=True)

    def _label_with_llm(self):
        config = Config(
            model_name="gemini-1.5-flash",
            max_tokens=0,
            max_depth=4,
            llm_provider="google",
            retry_attempts=2,
            retry_delay=0,
            success_delay=0,
            sleep_interval=0,
            max_workers=16
        )
        labeler = GraphLabeler(config, self.get_graph(), self.code)
        labeled_graph, _ = labeler.label_graph()
        attrs_to_set = {}
        for u, v, key, data in labeled_graph.edges(data=True, keys=True):
            if "domain_label" in data and data["domain_label"] in DOMAIN_EDGE_TYPES:
                attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES[data["domain_label"].upper()]
            else:
                attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES["NOT_INTERESTING"]


        self._logger.info(f"Successfully mapped {len(attrs_to_set)} predictions to edges.")
        self.set_domain_label(attrs_to_set, name="predicted_label")

    def _label_with_gat(self):
        encoded_graphs = self.model["encoder"].encode_graphs(
            [self.get_graph()], feature_to_encode="domain_label"
        )
        edge_predictions_for_response = []

        with torch.no_grad():
            i = len(encoded_graphs)
            graph_encoded = encoded_graphs[-1]
            output = self.model["model"](graph_encoded)
            preds = output["node_type_preds"]
            probabilities = torch.softmax(preds, dim=1)

            nx_G = self.get_graph()
            graph_edges_list = list(nx_G.edges(keys=True, data=True))

            if len(probabilities) != len(graph_edges_list):
                self._logger.warning(
                    f"Mismatch in graph {i}: Number of predictions ({len(probabilities)}) "
                    f"does not match number of edges ({len(graph_edges_list)}). "
                    "Cannot map labels to edges."
                )
                return

            prob_model_start = torch.tensor(
                [0, 0.4, 0.3, 0, 0.3], device=preds.device
            )
            prob_model_end = torch.tensor(
                [0.25, 0.15, 0.20, 0.15, 0.25], device=preds.device
            )
            in_degrees = nx_G.in_degree()
            out_degrees = nx_G.out_degree()
            start_mask = torch.tensor(
                [in_degrees[u] == 0 for u, v, k, d in graph_edges_list],
                dtype=torch.bool,
            )
            end_mask = torch.tensor(
                [out_degrees[v] == 0 for u, v, k, d in graph_edges_list],
                dtype=torch.bool,
            )
            end_mask &= ~start_mask
            start_mask = start_mask.to(preds.device)
            end_mask = end_mask.to(preds.device)
            probabilities[start_mask] *= prob_model_start
            probabilities[end_mask] *= prob_model_end

            labels = torch.argmax(probabilities, dim=1)
            self._logger.info(f"Graph {i}: Predicted {len(labels)} edge domain labels.")
            predicted_labels = labels.tolist()

            edge_keys = [(u, v, key) for u, v, key, data in graph_edges_list]
            attrs_to_set = dict(zip(edge_keys, predicted_labels))
            self.set_domain_label(attrs_to_set, name="predicted_label")

            self._logger.info(
                f"Successfully mapped {len(labels)} predictions to edges in graph {i}."
            )

    def _postprocess(self):
        G = self._current_state._G
        
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
            self.set_domain_node_label(node_type_updates, name="node_type")
            
        if edge_domain_updates:
            self.set_domain_label(edge_domain_updates, name="domain_label")
            self.set_domain_label(edge_numeric_updates, name="predicted_label") 
            self.set_domain_label(edge_numeric_updates, name="edge_type")

    def to_json(self) -> str:
        G = self._current_state._G
        V = nx.DiGraph()
        
        PRIMARY_ANCHORS = {DOMAIN_NODE_TYPES["DATASET"], DOMAIN_NODE_TYPES["MODEL"]}
        AUX_ANCHORS = {DOMAIN_NODE_TYPES["LIBRARY"], DOMAIN_NODE_TYPES["LITERAL"]}
        ALL_ANCHORS = PRIMARY_ANCHORS | AUX_ANCHORS

        anchor_nodes = set()
        
        for n, data in G.nodes(data=True):
            if data.get("node_type") in ALL_ANCHORS:
                anchor_nodes.add(n)
                V.add_node(n, **data)
                V.nodes[n]["transform_history"] = []
                V.nodes[n]["base_inputs"] = [] # Stores tuples: (priority, string_value)

        for start_node in anchor_nodes:
            visited = set()
            queue = [(start_node, [])]
            
            while queue:
                curr, path = queue.pop(0)
                
                for nxt in G.successors(curr):
                    if nxt in visited:
                        continue
                        
                    edge_dict = G[curr][nxt]
                    first_edge_data = next(iter(edge_dict.values()))
                    
                    edge_code = first_edge_data.get("code", "")
                    node_label = G.nodes[nxt].get("label", str(nxt))
                    
                    step_info = edge_code if edge_code else node_label
                    
                    if nxt in anchor_nodes:
                        new_ops = path + [step_info] if step_info else path
                        edge_type = first_edge_data.get("edge_type", DOMAIN_EDGE_TYPES.get("DATA_TRANSFORM", 2))
                        pred_label = first_edge_data.get("predicted_label", DOMAIN_EDGE_TYPES.get("DATA_TRANSFORM", 2))
                        
                        V.add_edge(start_node, nxt, operations=new_ops, edge_type=edge_type, predicted_label=pred_label)
                    else:
                        visited.add(nxt)
                        new_path = path + [step_info] if step_info else path
                        queue.append((nxt, new_path))

        aux_nodes_in_V = [n for n in V.nodes if V.nodes[n].get("node_type") in AUX_ANCHORS]
        for aux in aux_nodes_in_V:
            aux_type = V.nodes[aux].get("node_type")
            aux_label = V.nodes[aux].get("label", str(aux))
            
            for v in list(V.successors(aux)):
                if V.nodes[v].get("node_type") in PRIMARY_ANCHORS:
                    edge_data = V.edges[aux, v]
                    ops = edge_data.get("operations", [])
                    edge_str = " -> ".join([str(o) for o in ops if o])
                    
                    if aux_type == DOMAIN_NODE_TYPES["LIBRARY"]:
                        fmt_val = f"{aux_label}.{edge_str}" if edge_str else aux_label
                        V.nodes[v]["base_inputs"].append((0, fmt_val))
                    else:
                        fmt_val = edge_str if edge_str else aux_label
                        V.nodes[v]["base_inputs"].append((1, f'"{fmt_val}"'))
                        
            V.remove_node(aux)

        changed = True
        while changed:
            changed = False
            for u in list(V.nodes):
                if u not in V: 
                    continue
                
                if V.nodes[u].get("node_type") != DOMAIN_NODE_TYPES["DATASET"]:
                    continue
                
                dataset_successors = [s for s in V.successors(u) if V.nodes[s].get("node_type") == DOMAIN_NODE_TYPES["DATASET"]]
                
                if len(dataset_successors) == 1:
                    v = dataset_successors[0]
                    
                    dataset_predecessors = [p for p in V.predecessors(v) if V.nodes[p].get("node_type") == DOMAIN_NODE_TYPES["DATASET"]]
                    
                    if len(dataset_predecessors) == 1 and u != v:
                        ops = V.edges[u, v].get("operations", [])
                        v_label = V.nodes[v].get("label", str(v))
                        
                        step_desc = " -> ".join([str(o) for o in ops if o]) if ops else "Transform"
                        
                        raw_aux = V.nodes[v].get("base_inputs", [])
                        sorted_aux = [item[1] for item in sorted(raw_aux, key=lambda x: x[0])]
                        aux_str = f" [Inputs: {', '.join(sorted_aux)}]" if sorted_aux else ""
                        
                        step = {
                            "target_node": v_label,
                            "operation": step_desc + aux_str,
                            "transform_code": "" 
                        }
                        
                        V.nodes[u]["transform_history"].append(step)
                        V.nodes[u]["transform_history"].extend(V.nodes[v].get("transform_history", []))
                        
                        for v_succ in list(V.successors(v)):
                            edge_data = V.edges[v, v_succ]
                            V.add_edge(u, v_succ, **edge_data)
                            
                        V.remove_node(v)
                        changed = True
                        break

        elements = []
        for n, data in V.nodes(data=True):
            raw_inputs = data.get("base_inputs", [])
            sorted_inputs = [item[1] for item in sorted(raw_inputs, key=lambda x: x[0])]
            base_inputs_str = ", ".join(sorted_inputs) if sorted_inputs else ""
            
            elements.append({
                "data": {
                    "id": str(n),
                    "label": data.get("label", str(n)),
                    "node_type": data.get("node_type", 0),
                    "code": data.get("code", ""),
                    "transform_history": data.get("transform_history", []),
                    "base_inputs": base_inputs_str
                }
            })
            
        for u, v, data in V.edges(data=True):
            ops = data.get("operations", [])
            clean_ops = [str(o) for o in ops if o]
            edge_lbl = " -> ".join(clean_ops) if clean_ops else data.get("label", "Transform")
            
            elements.append({
                "data": {
                    "source": str(u),
                    "target": str(v),
                    "edge_type": data.get("edge_type", 2), 
                    "label": edge_lbl,
                    "predicted_label": data.get("predicted_label", 2) 
                }
            })
            
        return json.dumps({"elements": elements})