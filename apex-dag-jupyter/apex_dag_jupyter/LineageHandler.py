import os
import json
import torch
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.label_notebooks.online_labeler import OnlineGraphLabeler as GraphLabeler
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES


class LineageHandler(APIHandler):
    def initialize(self, model, jupyter_server_app_config=None):
        self.model = model
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_results = {}

    @tornado.web.authenticated
    def post(self):
        try:
            input_data = self.get_json_body()
            code = input_data["code"]
            replace_dataflow = input_data["replaceDataflowInUDFs"]
            hightlight_relevant = input_data["highlightRelevantSubgraphs"]
            llm_classification = input_data["llmClassification"]

            dfg = DataFlowGraph(replace_dataflow=replace_dataflow)
            try:
                dfg.parse_code(code)
            except SyntaxError as e:
                self.log.error(f"SyntaxError: {e}", exc_info=True)
                result = {
                    "message": "Cannot process lineage due to a syntax error. Returning last successful result.",
                    "success": False,
                    "dataflow": self.last_analysis_results,
                }
                self.set_status(400)
                self.finish(json.dumps(result))
                return

            dfg.optimize()
            self.log.info(
                f"Optimized DFG graph has {len(dfg.get_graph().nodes)} nodes and {len(dfg.get_graph().edges)} edges."
            )
            if llm_classification:
                dfg = self._label_with_llm(dfg)
            else:
                dfg = self._label_with_gat(dfg)

            if hightlight_relevant:
                dfg.filter_relevant()
            dfg.optimize()

            self.last_analysis_results = dfg.to_json()
            result = {
                "message": "Processed dataflow successfully!",
                "success": True,
                "lineage_predictions": self.last_analysis_results,
            }
            self.finish(json.dumps(result))
        except Exception as e:
            self.log.error(f"An unexpected error occurred in LineageHandler: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({
                "message": "An internal server error occurred.",
                "success": False,
            }))

    def _label_with_llm(self, dfg: DataFlowGraph):
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
        labeler = GraphLabeler(config, dfg.get_graph(), dfg.code)
        labeled_graph, _ = labeler.label_graph()
        attrs_to_set = {}
        for u, v, key, data in labeled_graph.edges(data=True, keys=True):
            if "domain_label" in data and data["domain_label"] in DOMAIN_EDGE_TYPES:
                attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES[data["domain_label"].upper()]
            else:
                attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES["NOT_INTERESTING"]


        self.log.info(f"Successfully mapped {len(attrs_to_set)} predictions to edges.")
        dfg.set_domain_label(attrs_to_set, name="predicted_label")
        return dfg


    def _label_with_gat(self, dfg: DataFlowGraph):
        encoded_graphs = self.model["encoder"].encode_graphs(
            [dfg.get_graph()], feature_to_encode="domain_label"
        )
        edge_predictions_for_response = []

        with torch.no_grad():
            i = len(encoded_graphs)
            graph_encoded = encoded_graphs[-1]
            output = self.model["model"](graph_encoded)
            preds = output["node_type_preds"]
            probabilities = torch.softmax(preds, dim=1)

            nx_G = dfg.get_graph()
            graph_edges_list = list(nx_G.edges(keys=True, data=True))

            if len(probabilities) != len(graph_edges_list):
                self.log.warning(
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
            self.log.info(f"Graph {i}: Predicted {len(labels)} edge domain labels.")
            predicted_labels = labels.tolist()

            edge_keys = [(u, v, key) for u, v, key, data in graph_edges_list]
            attrs_to_set = dict(zip(edge_keys, predicted_labels))
            dfg.set_domain_label(attrs_to_set, name="predicted_label")

            self.log.info(
                f"Successfully mapped {len(labels)} predictions to edges in graph {i}."
            )

            return dfg

    def data_received(self, chunk):
        """Override to silence Tornado abstract method warning."""
        pass
