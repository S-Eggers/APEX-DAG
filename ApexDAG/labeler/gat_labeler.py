import torch
import logging
from ApexDAG.labeler.edge_labeler import EdgeLabeler
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph
from ApexDAG.labeler.heuristics import DegreeBasedHeuristic

logger = logging.getLogger(__name__)

class GATLabeler(EdgeLabeler):
    def __init__(self, model_dict: dict):
        """
        Expects a dictionary containing the 'model' (MultiTaskGATv1) and 'encoder'.
        """
        self.encoder = model_dict["encoder"]
        self.model = model_dict["model"]
        
        self.device = next(self.model.parameters()).device
        self.heuristic = DegreeBasedHeuristic(device=self.device)

    def apply_labels(self, graph: PythonDataFlowGraph) -> None:
        nx_G = graph.get_graph()
        graph_edges_list = list(nx_G.edges(keys=True, data=True))

        if not graph_edges_list:
            return

        encoded_graphs = self.encoder.encode_graphs(
            [nx_G], feature_to_encode="domain_label"
        )
        graph_encoded = encoded_graphs[-1].to(self.device)

        with torch.no_grad():
            output = self.model(graph_encoded)
            probabilities = output["node_type_preds"] 

            if len(probabilities) != len(graph_edges_list):
                logger.error(
                    f"Prediction mismatch: {len(probabilities)} predictions vs "
                    f"{len(graph_edges_list)} edges. Cannot map labels."
                )
                return

            probabilities = self.heuristic.apply(probabilities, nx_G, graph_edges_list)
            labels = torch.argmax(probabilities, dim=1)
            predicted_labels = labels.tolist()

        edge_keys = [(u, v, key) for u, v, key, data in graph_edges_list]
        attrs_to_set = dict(zip(edge_keys, predicted_labels))
        graph.set_domain_label(attrs_to_set, name="predicted_label")