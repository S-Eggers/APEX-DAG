import json
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.sca.py_ast_graph import PythonASTGraph as ASTGraph


class ASTHandler(APIHandler):
    def initialize(self, model, jupyter_server_app_config=None):
        self.model = model
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_results = {}

    @tornado.web.authenticated
    def post(self):
        input_data = self.get_json_body()
        code = input_data["code"]
        replace_dataflow = input_data["replaceDataflowInUDFs"]
        hightlight_relevant = input_data["highlightRelevantSubgraphs"]
        ast = ASTGraph()
        try:
            ast.parse_code(code)
        except SyntaxError as e:
            print(f"SyntaxError: {e}")
            result = {
                "message": "Cannot process AST! Returning last successful result.",
                "success": False,
                "ast_graph": self.last_analysis_results,
            }
            self.finish(json.dumps(result))
        else:

   
            graph_json = ast.to_json()
            self.last_analysis_results = graph_json
            result = {
                "message": "Processed AST successfully!",
                "success": True,
                "ast_graph": graph_json,
            }
            self.finish(json.dumps(result))

    def data_received(self, chunk):
        """Override to silence Tornado abstract method warning."""
        pass
