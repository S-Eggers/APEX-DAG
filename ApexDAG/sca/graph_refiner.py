from ApexDAG.sca import (
    DOMAIN_EDGE_TYPES,
    DOMAIN_NODE_TYPES,
    NODE_TYPES,
    REVERSE_DOMAIN_EDGE_TYPES,
)
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph


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
        known_hyperparams = set()

        # RULE 1: BASE TRUTH
        for n, data in G.nodes(data=True):
            if data.get("node_type") == DOMAIN_NODE_TYPES["DATASET"]:
                known_datasets.add(n)
            elif data.get("node_type") == DOMAIN_NODE_TYPES["MODEL"]:
                known_models.add(n)

        # 2. STATIC SEEDING
        for u, v, key, data in G.edges(keys=True, data=True):
            domain_label = data.get("predicted_label", "")
            if domain_label not in REVERSE_DOMAIN_EDGE_TYPES:
                continue

            edge_type_name = REVERSE_DOMAIN_EDGE_TYPES[domain_label]

            # RULE 2: EXTRACTION SINK
            if edge_type_name == "DATA_IMPORT_EXTRACTION":
                known_datasets.add(v)

            # RULE 3: MODEL TARGET
            elif edge_type_name == "MODEL_OPERATION":
                known_models.add(v)

        # RULE 5: DATA PROPAGATION (DYNAMIC TAINT)
        changed = True
        while changed:
            changed = False
            for u, v, key, data in G.edges(keys=True, data=True):

                # STRUCTURAL INTERMEDIATE BYPASS
                if G.nodes[v].get("node_type") == NODE_TYPES["INTERMEDIATE"]:
                    if u in known_datasets and v not in known_datasets:
                        known_datasets.add(v)
                        changed = True
                        continue
                    elif u in known_models and v not in known_models and v not in known_datasets:
                        known_models.add(v)
                        changed = True
                        continue

                # RULE 5.2: STANDARD SEMANTIC PROPAGATION
                domain_label = data.get("predicted_label", "")
                if domain_label not in REVERSE_DOMAIN_EDGE_TYPES:
                    continue

                edge_type_name = REVERSE_DOMAIN_EDGE_TYPES[domain_label]

                if edge_type_name == "DATA_TRANSFORM":
                    if u in known_datasets and v not in known_datasets:
                        if v not in known_models:
                            known_datasets.add(v)
                            changed = True

                elif edge_type_name == "EDA":
                    if u in known_datasets and v not in known_datasets:
                        known_datasets.add(v)
                        changed = True

        for u, v, key, data in G.edges(keys=True, data=True):
            is_u_literal = G.nodes[u].get("node_type") == NODE_TYPES["LITERAL"]

            if is_u_literal:
                edge_type_name = REVERSE_DOMAIN_EDGE_TYPES.get(data.get("predicted_label", ""), "")

                # RULE 6: IMPORT PATH RESOLUTION
                if v in known_datasets and edge_type_name != "DATA_IMPORT_EXTRACTION":
                    edge_domain_updates[(u, v, key)] = "DATA_IMPORT_EXTRACTION"
                    edge_numeric_updates[(u, v, key)] = DOMAIN_EDGE_TYPES["DATA_IMPORT_EXTRACTION"]

                # RULE 7: HYPERPARAMETER RESOLUTION
                elif v in known_models or edge_type_name == "MODEL_OPERATION":
                    known_hyperparams.add(u)

        # RULE 8: MUTUAL EXCLUSION
        for n in known_datasets:
            if G.nodes[n].get("node_type") in UPGRADABLE_NODE_TYPES:
                node_type_updates[n] = DOMAIN_NODE_TYPES["DATASET"]

        for n in known_models:
            if n not in node_type_updates and G.nodes[n].get("node_type") in UPGRADABLE_NODE_TYPES:
                node_type_updates[n] = DOMAIN_NODE_TYPES["MODEL"]

        for n, data in G.nodes(data=True):
            if data.get("node_type") == NODE_TYPES["LITERAL"]:
                if n in known_hyperparams:
                    node_type_updates[n] = DOMAIN_NODE_TYPES["HYPERPARAMETER"]
                else:
                    node_type_updates[n] = DOMAIN_NODE_TYPES["LITERAL"]

        # 9. EXECUTION
        if node_type_updates:
            graph.set_domain_node_label(node_type_updates, name="node_type")

        if edge_domain_updates:
            graph.set_domain_label(edge_domain_updates, name="domain_label")
            graph.set_domain_label(edge_numeric_updates, name="predicted_label")
            graph.set_domain_label(edge_numeric_updates, name="edge_type")
