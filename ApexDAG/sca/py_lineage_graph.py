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