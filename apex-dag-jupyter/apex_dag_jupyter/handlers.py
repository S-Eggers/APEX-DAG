import json
import time
import torch
import tornado
import networkx as nx
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join

from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.sca.constants import REVERSE_DOMAIN_EDGE_TYPES


class DataflowHandler(APIHandler):
    def initialize(self, model, jupyter_server_app_config=None):
        self.model = model
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_results = {}

    @tornado.web.authenticated
    def post(self):
        input_data = self.get_json_body()
        code = input_data["code"]
        dfg = DataFlowGraph()
        try:
            dfg.parse_code(code)
        except SyntaxError as e:
            print(f"SyntaxError: {e}")
            result = {
                "message": "Cannot process dataflow! Returning last successful result.",
                "success": False,
                "dataflow": self.last_analysis_results
            }
            self.finish(json.dumps(result))
        else:
            dfg.optimize()
            graph_json = dfg.to_json()
            self.last_analysis_results = graph_json
            result = {
                "message": "Processed dataflow successfully!",
                "success": True,
                "dataflow": graph_json
            }
            self.finish(json.dumps(result))

    def data_received(self, chunk):
        """Override to silence Tornado abstract method warning."""
        pass


class LineageHandler(APIHandler):
    def initialize(self, model, jupyter_server_app_config=None):
        self.model = model
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_time = 0
        self.last_analysis_results = {}


    @tornado.web.authenticated
    def post(self):
        input_data = self.get_json_body()
        code = input_data["code"]

        if time.time() - self.last_analysis_time < 200:
            result = self.last_analysis_results
            print("Reusing results")
            self.finish(json.dumps(result))
            return
        self.last_analysis_time = time.time()

        dfg = DataFlowGraph()
        try:
            dfg.parse_code(code)
        except SyntaxError as e:
            print(f"SyntaxError: {e}")
            result = {
                "message": "Cannot process dataflow!",
                "success": False,
                "dataflow": {}
            }

            self.finish(json.dumps(result))
        else:
            dfg.optimize()
            encoded_graphs = self.model["encoder"].encode_graphs([dfg.get_graph()], feature_to_encode="domain_label")
            edge_predictions_for_response = []

            with torch.no_grad():
                for i, graph_encoded in enumerate(encoded_graphs):
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

                    # Apply probability model / heuristics
                    prob_model_start = torch.tensor([0, 0.5, 0, 0, 0.5], device=preds.device)
                    prob_model_end = torch.tensor([0.25, 0.15, 0.20, 0.15, 0.25], device=preds.device)
                    in_degrees = nx_G.in_degree()
                    out_degrees = nx_G.out_degree()
                    start_mask = torch.tensor([in_degrees[u] == 0 for u, v, k, d in graph_edges_list], dtype=torch.bool)
                    end_mask = torch.tensor([out_degrees[v] == 0 for u, v, k, d in graph_edges_list], dtype=torch.bool)
                    end_mask &= ~start_mask
                    start_mask = start_mask.to(preds.device)
                    end_mask = end_mask.to(preds.device)
                    probabilities[start_mask] *= prob_model_start
                    probabilities[end_mask] *= prob_model_end

                    # Get final labels 
                    labels = torch.argmax(probabilities, dim=1)
                    self.log.info(f"Graph {i}: Predicted {len(labels)} edge domain labels.")
                    predicted_label_names = [REVERSE_DOMAIN_EDGE_TYPES[l.item()] for l in labels]

                    # Update graph
                    edge_keys = [(u, v, key) for u, v, key, data in graph_edges_list]
                    attrs_to_set = dict(zip(edge_keys, predicted_label_names))
                    nx.set_edge_attributes(nx_G, attrs_to_set, name="predicted_domain_label")

                    # Build response
                    edge_predictions_for_response = [
                        {
                            "source": str(u),
                            "target": str(v),
                            "key": key,
                            "code": edge_data.get("code", ""),
                            "predicted_domain": predicted_label_names[edge_idx]
                        }
                        for edge_idx, (u, v, key, edge_data) in enumerate(graph_edges_list)
                    ]

                    self.log.info(f"Successfully mapped {len(labels)} predictions to edges in graph {i}.")

            result = {
                "message": "Processed dataflow successfully!",
                "success": True,
                "lineage_predictions": dfg.to_json(), # Include mapped predictions
            }
            print(result)
            self.finish(json.dumps(result))

    def data_received(self, chunk):
        """Override to silence Tornado abstract method warning."""
        pass


def setup_handlers(web_app, model_instance, jupyter_server_app_config=None):
    host_pattern = ".*$"

    base_url = web_app.settings["base_url"]

    dataflow_pattern = url_path_join(base_url, "apex-dag", "dataflow")
    lineage_pattern = url_path_join(base_url, "apex-dag", "lineage")
    handlers = [
        (dataflow_pattern, DataflowHandler, dict(model=model_instance, jupyter_server_app_config=jupyter_server_app_config)),
        (lineage_pattern, LineageHandler, dict(model=model_instance, jupyter_server_app_config=jupyter_server_app_config)),
    ]

    web_app.add_handlers(host_pattern, handlers)
