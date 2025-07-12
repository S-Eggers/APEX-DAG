import json
import time
import torch
import tornado
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

            with torch.no_grad():
                for i, graph_encoded in enumerate(encoded_graphs):
                    output = self.model["model"](graph_encoded)
                    labels = torch.argmax(output['node_type_preds'], dim=1)
                    labels_names = [REVERSE_DOMAIN_EDGE_TYPES[label.item()] for label in labels]

                    self.log.info(f"Graph {i}: Predicted {len(labels)} edge domain labels.")

                    nx_G = dfg.get_graph()
                    graph_edges_list = list(nx_G.edges(keys=True, data=True))

                    if len(labels) == len(graph_edges_list):
                        for edge_idx, (u, v, key, edge_data) in enumerate(graph_edges_list):
                            predicted_label_int = labels[edge_idx].item()
                            predicted_label_name = REVERSE_DOMAIN_EDGE_TYPES[predicted_label_int]
                            dfg.set_domain_label(u, v, key, predicted_label_int)
                            
                            #for edge, label in zip(nx_G.edges, labels_names):
                            #    print(f"Edge {edge} [{label}]: {edge[0]} - {nx_G.edges._adjdict[edge[0]][edge[1]]} -> {edge[1] }")

                        self.log.info(f"Successfully mapped {len(labels)} predictions to edges in graph {i}.")
                    else:
                        self.log.warning(
                            f"Mismatch in graph {i}: Number of predictions ({len(labels)}) "
                            f"does not match number of edges ({len(graph_edges_list)}). "
                            "Cannot map labels to edges."
                        )

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
