import json
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.pipeline.dataflow_pipeline_factory import DataflowPipelineFactory

class DataflowHandler(APIHandler):
    def initialize(self, jupyter_server_app_config=None):
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_results = {}

    @tornado.web.authenticated
    def post(self):
        try:
            input_data = self.get_json_body()
            code = input_data.get("code", "")
            
            pipeline = DataflowPipelineFactory.create(input_data)
            
            try:
                analysis_results = pipeline.execute(code)
            except SyntaxError as e:
                self.log.error(f"SyntaxError during dataflow AST parsing: {e}", exc_info=True)
                self.set_status(400)
                self.finish(json.dumps({
                    "message": "Cannot process dataflow due to a syntax error. Returning last successful result.",
                    "success": False,
                    "dataflow": self.last_analysis_results,
                }))
                return

            self.last_analysis_results = analysis_results
            self.finish(json.dumps({
                "message": "Processed dataflow successfully!",
                "success": True,
                "dataflow": analysis_results,
            }))

        except Exception as e:
            self.log.error(f"Unexpected error in DataflowHandler: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({
                "message": "An internal server error occurred.",
                "success": False,
            }))

    def data_received(self, chunk):
        pass