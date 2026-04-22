from typing import Dict, Any
import networkx as nx

from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph
from ApexDAG.sca import (
    NODE_TYPES,
    EDGE_TYPES,
    DOMAIN_EDGE_TYPES,
    DOMAIN_NODE_TYPES,
)


class LineageSerializer:
    def to_dict(self, graph: PythonDataFlowGraph) -> Dict[str, Any]:
        G = graph.get_graph()
        V = nx.DiGraph()
        
        PRIMARY_ANCHORS = {DOMAIN_NODE_TYPES["DATASET"], DOMAIN_NODE_TYPES["MODEL"]}
        AUX_ANCHORS = {DOMAIN_NODE_TYPES["LIBRARY"], DOMAIN_NODE_TYPES["LITERAL"]}
        ALL_ANCHORS = PRIMARY_ANCHORS | AUX_ANCHORS

        anchor_nodes = set()
        
        # 1. Initialize anchor nodes dynamically (Preserves cell_id on nodes)
        for n, data in G.nodes(data=True):
            if data.get("node_type") in ALL_ANCHORS:
                anchor_nodes.add(n)
                V.add_node(n, **data)
                V.nodes[n]["transform_history"] = []
                V.nodes[n]["base_inputs"] = []

        # 2. Traverse and build edges
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
                        
                        inherited_cell_id = first_edge_data.get("cell_id", G.nodes[nxt].get("cell_id", "unknown_cell"))
                        
                        V.add_edge(
                            start_node, 
                            nxt, 
                            operations=new_ops, 
                            edge_type=edge_type, 
                            predicted_label=pred_label,
                            cell_id=inherited_cell_id
                        )
                    else:
                        visited.add(nxt)
                        new_path = path + [step_info] if step_info else path
                        queue.append((nxt, new_path))

        # 3. Collapse Aux Nodes
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

        # 4. Collapse Linear Datasets
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
            
            payload = {
                "id": str(n),
                "label": data.get("label", str(n)),
                "node_type": 0,
                "transform_history": data.get("transform_history", []),
                "base_inputs": base_inputs_str
            }
            
            safe_data = {k: v for k, v in data.items() if k not in ("id", "base_inputs", "transform_history")}
            payload.update(safe_data)
            
            elements.append({"data": payload})
            
        for u, v, data in V.edges(data=True):
            ops = data.get("operations", [])
            clean_ops = [str(o) for o in ops if o]
            edge_lbl = " -> ".join(clean_ops) if clean_ops else data.get("label", "Transform")
            
            payload = {
                "id": f"edge_{u}_{v}",
                "source": str(u),
                "target": str(v),
                "edge_type": 2, 
                "label": edge_lbl,
                "predicted_label": data.get("predicted_label", 2) 
            }
            
            safe_data = {k: val for k, val in data.items() if k not in ("id", "operations")}
            payload.update(safe_data)
            
            elements.append({"data": payload})
            
        return {"elements": elements}