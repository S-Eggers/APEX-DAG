import json
import time
import torch
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph


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
                    prob_model_start = torch.tensor([0, 0.6, 0, 0, 0.4], device=preds.device)
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
                    predicted_labels = labels.tolist()

                    # Update graph
                    edge_keys = [(u, v, key) for u, v, key, data in graph_edges_list]
                    attrs_to_set = dict(zip(edge_keys, predicted_labels))
                    dfg.set_domain_label(attrs_to_set, name="predicted_label")

                    self.log.info(f"Successfully mapped {len(labels)} predictions to edges in graph {i}.")

            result = {
                "message": "Processed dataflow successfully!",
                "success": True,
                "lineage_predictions": dfg.to_json(), # Include mapped predictions
            }
            self.finish(json.dumps(result))

    def data_received(self, chunk):
        """Override to silence Tornado abstract method warning."""
        pass

