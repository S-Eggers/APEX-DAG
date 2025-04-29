import json

import tornado
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join

from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph


class DataflowHandler(APIHandler):
    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)

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
                "message": "Cannot process dataflow!",
                "success": False,
                "dataflow": {}
            }
            self.finish(json.dumps(result))
        else:
            dfg.optimize()
            graph_json = dfg.to_json()

            # Do your Dataflow processing here
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
    @tornado.web.authenticated
    def post(self):
        input_data = self.get_json_body()

        # Do your Lineage processing here
        result = {
            "message": "Processed lineage successfully!",
            "input": input_data
        }

        self.finish(json.dumps(result))

    def data_received(self, chunk):
        """Override to silence Tornado abstract method warning."""
        pass


def setup_handlers(web_app):
    host_pattern = ".*$"

    base_url = web_app.settings["base_url"]
    # Prepend the base_url so that it works in a JupyterHub setting

    dataflow_pattern = url_path_join(base_url, "apex-dag", "dataflow")
    lineage_pattern = url_path_join(base_url, "apex-dag", "lineage")
    handlers = [
        (dataflow_pattern, DataflowHandler),
        (lineage_pattern, LineageHandler),
    ]

    web_app.add_handlers(host_pattern, handlers)
