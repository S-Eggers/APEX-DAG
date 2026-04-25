import json
from pathlib import Path

import tornado
from ApexDAG.pipeline.labeling_pipeline_factory import LabelingPipelineFactory

from .ApexDAGBaseHandler import ApexDAGBaseHandler


class LabelingPredictHandler(ApexDAGBaseHandler):
    @property
    def response_key(self) -> str:
        return "predictions"

    def create_pipeline(self, input_data: dict):
        return LabelingPipelineFactory.create(input_data, self.model)

    @tornado.web.authenticated
    def post(self):
        input_data = self.get_json_body()
        requested_filename = input_data.get("filename", "")

        if requested_filename:
            safe_base_dir = Path.home() / ".apexdag" / "annotations"
            secure_filename = Path(requested_filename).name

            if secure_filename.endswith('.ipynb'):
                json_filename = secure_filename.replace('.ipynb', '.json')
            else:
                json_filename = f"{secure_filename}.json"

            target_path = (safe_base_dir / json_filename).resolve()

            if target_path.exists() and target_path.is_relative_to(safe_base_dir):
                self.log.info(f"CACHE HIT: Loading JSON annotation for {secure_filename}")
                try:
                    with open(target_path, encoding="utf-8") as f:
                        cached_elements = json.load(f)

                    self.finish(json.dumps({
                        "message": "Loaded existing annotations from disk.",
                        "success": True,
                        self.response_key: {"elements": cached_elements}
                    }))
                    return

                except Exception as e:
                    self.log.error(f"Failed to load JSON: {e}. Falling back to ML.", exc_info=True)

        self.log.info("CACHE MISS: Executing ML prediction pipeline.")
        super().post()
