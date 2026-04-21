import json
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.pipeline.environment_pipeline_factory import EnvironmentPipelineFactory


class EnvironmentHandler(APIHandler):
    def initialize(self, jupyter_server_app_config=None):
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_results = {}

    @tornado.web.authenticated
    def post(self):
        try:
            input_data = self.get_json_body()
            code = input_data.get("code", "")

            pipeline = EnvironmentPipelineFactory.create()

            try:
                analysis_results = pipeline.execute(code)
            except SyntaxError as e:
                self.log.error(f"SyntaxError in EnvironmentHandler AST parsing: {e}", exc_info=True)
                self.set_status(400)
                self.finish(json.dumps({
                    "message": "Syntax error in notebook. Returning last valid state.",
                    "success": False,
                    "environment_data": self.last_analysis_results
                }))
                return

            self.last_analysis_results = analysis_results
            self.finish(json.dumps({
                "message": "Environment analyzed successfully.",
                "success": True,
                "environment_data": analysis_results
            }))

        except Exception as e:
            self.log.error(f"Unexpected error in EnvironmentHandler: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({
                "message": "Internal server error.",
                "success": False
            }))

    def data_received(self, chunk):
        pass