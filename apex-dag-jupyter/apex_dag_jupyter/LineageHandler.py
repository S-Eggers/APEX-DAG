import json
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.sca.py_lineage_graph import PythonLineageGraph as LineageGraph


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

            lgraph = LineageGraph(
                model=self.model, 
                use_llm_backend=llm_classification, 
                highlight_relevant=hightlight_relevant, 
                replace_dataflow=replace_dataflow
            )
            try:
                lgraph.parse_code(code)
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

            self.last_analysis_results = lgraph.to_json()
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

    def data_received(self, chunk):
        """Override to silence Tornado abstract method warning."""
        pass
