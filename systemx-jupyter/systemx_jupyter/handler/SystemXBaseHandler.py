import json

import tornado
from jupyter_server.base.handlers import APIHandler

class BackendNotAvailableError(Exception):
    """Raised when a requested NN backend is not loaded in this server process."""

LEARNED_MODEL_FAMILIES = ("hgt", "mlp", "xgboost")
FEATURE_PRESETS = ("standard", "all", "emb_only", "api_lib", "struct_only")

class SystemXBaseHandler(APIHandler):
    """Abstract base handler for all SystemX pipeline executions."""

    def resolve_learned_labeler(self, backend: str, preset: str, variant_key: str | None = None) -> object:
        """Return the loaded labeler for a family and feature-preset selection, or an explicit variant key."""
        if variant_key:
            labeler = self.models.get(variant_key)
            if labeler is None:
                raise BackendNotAvailableError(
                    f"Model variant '{variant_key}' is not loaded. Retrain it, or pick a different "
                    f"variant (the manifest.json must contain key '{variant_key}')."
                )
            return labeler

        if backend not in LEARNED_MODEL_FAMILIES:
            raise BackendNotAvailableError(
                f"Unknown model '{backend}'. Expected one of {LEARNED_MODEL_FAMILIES} or 'vamsa_static'."
            )
        if preset not in FEATURE_PRESETS:
            raise BackendNotAvailableError(f"Unknown feature preset '{preset}'. Expected one of {FEATURE_PRESETS}.")

        key = f"{backend}_{preset}"
        labeler = self.models.get(key)
        if labeler is None:
            raise BackendNotAvailableError(
                f"Backend '{key}' is not loaded. Train it (e.g. "
                f"`python -m SystemX.experiment.ablation.train_all`) and point "
                f"c.SystemXConfig.v2_checkpoints_dir at the output directory "
                f"(the manifest.json must contain key '{key}')."
            )
        return labeler

    def initialize(
        self,
        models: dict | None = None,
        jupyter_server_app_config: dict | None = None,
    ) -> None:
        self.models: dict = models or {}
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_results = {}

    @property
    def response_key(self) -> str:
        """The JSON key used to return the graph data (e.g., 'dataflow', 'ast_graph')."""
        raise NotImplementedError("Subclasses must define the response key.")

    def create_pipeline(self, input_data: dict) -> None:
        """Instantiate and return the specific PipelineFactory object."""
        raise NotImplementedError("Subclasses must implement create_pipeline.")

    def postprocess_results(self, analysis_results: object, input_data: dict) -> object:
        """Transform the serialized graph before it is returned; a no-op by default."""
        return analysis_results

    @tornado.web.authenticated
    def post(self) -> None:
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

            try:
                pipeline = self.create_pipeline(input_data)
            except BackendNotAvailableError as e:
                self.log.warning("Backend not available: %s", e)
                self.set_status(200)
                self.finish(
                    json.dumps(
                        {
                            "message": str(e),
                            "success": False,
                            "backend_error": str(e),
                            self.response_key: {},
                        }
                    )
                )
                return

            try:
                analysis_results = pipeline.execute(cells)
            except SyntaxError as e:
                self.log.error(f"SyntaxError in {self.__class__.__name__}: {e}", exc_info=True)
                self.set_status(400)
                self.finish(
                    json.dumps(
                        {
                            "message": f"Syntax error: {e!s}",
                            "success": False,
                            self.response_key: self.last_analysis_results,
                        }
                    )
                )
                return

            analysis_results = self.postprocess_results(analysis_results, input_data)
            self.last_analysis_results = analysis_results
            self.finish(
                json.dumps(
                    {
                        "message": "Processed successfully!",
                        "success": True,
                        self.response_key: analysis_results,
                    }
                )
            )

        except Exception as e:
            self.log.error(f"Unexpected error in {self.__class__.__name__}: {e}", exc_info=True)
            self.set_status(500)
            self.finish(
                json.dumps(
                    {
                        "message": "An internal server error occurred.",
                        "success": False,
                    }
                )
            )

    def data_received(self, chunk: bytes) -> None:
        pass
