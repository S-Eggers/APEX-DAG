import json

import tornado
from jupyter_server.base.handlers import APIHandler


class ApexDAGBaseHandler(APIHandler):
    """
    Abstract base handler for all ApexDAG pipeline executions.
    """

    def initialize(self, model=None, jupyter_server_app_config=None):
        self.model = model
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_results = {}

    @property
    def response_key(self) -> str:
        """The JSON key used to return the graph data (e.g., 'dataflow', 'ast_graph')."""
        raise NotImplementedError("Subclasses must define the response key.")

    def create_pipeline(self, input_data: dict):
        """Instantiate and return the specific PipelineFactory object."""
        raise NotImplementedError("Subclasses must implement create_pipeline.")

    @tornado.web.authenticated
    def post(self):
        try:
            input_data = self.get_json_body()

            cells_payload = input_data.get("cells")
            code_payload = input_data.get("code", "")

            if isinstance(cells_payload, list) and len(cells_payload) > 0:
                cells = cells_payload
            elif isinstance(code_payload, list):
                self.log.warning("Frontend sent cell array under the 'code' key. Coercing to 'cells'.")
                cells = code_payload
            else:
                self.log.warning("Legacy string payload received. Applying fallback cell_id.")
                cells = [{"cell_id": "legacy_fallback", "source": str(code_payload)}]

            pipeline = self.create_pipeline(input_data)

            try:
                analysis_results = pipeline.execute(cells)
            except SyntaxError as e:
                self.log.error(f"SyntaxError in {self.__class__.__name__}: {e}", exc_info=True)
                self.set_status(400)
                self.finish(json.dumps({
                    "message": f"Syntax error: {e!s}",
                    "success": False,
                    self.response_key: self.last_analysis_results,
                }))
                return

            self.last_analysis_results = analysis_results
            self.finish(json.dumps({
                "message": "Processed successfully!",
                "success": True,
                self.response_key: analysis_results,
            }))

        except Exception as e:
            self.log.error(f"Unexpected error in {self.__class__.__name__}: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({
                "message": "An internal server error occurred.",
                "success": False,
            }))

    def data_received(self, chunk):
        pass
