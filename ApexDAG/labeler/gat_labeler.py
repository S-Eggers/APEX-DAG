import torch
import networkx as nx

from ApexDAG.labeler.edge_labeler import EdgeLabeler
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph


class GATLabeler(EdgeLabeler):
    def __init__(self, model):
        self.model = model

    def apply_labels(self, graph: PythonDataFlowGraph) -> None:
        encoded_graphs = self.model["encoder"].encode_graphs(
            [graph.get_graph()], feature_to_encode="domain_label"
        )
        edge_predictions_for_response = []

        with torch.no_grad():
            i = len(encoded_graphs)
            graph_encoded = encoded_graphs[-1]
            output = self.model["model"](graph_encoded)
            preds = output["node_type_preds"]
            probabilities = torch.softmax(preds, dim=1)

            nx_G = graph.get_graph()
            graph_edges_list = list(nx_G.edges(keys=True, data=True))

            if len(probabilities) != len(graph_edges_list):
                print(
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
            predicted_labels = labels.tolist()

            edge_keys = [(u, v, key) for u, v, key, data in graph_edges_list]
            attrs_to_set = dict(zip(edge_keys, predicted_labels))
            graph.set_domain_label(attrs_to_set, name="predicted_label")