import json
import time
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph


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