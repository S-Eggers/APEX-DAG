import json
from pathlib import Path

import tornado
from SystemX.pipeline.dataflow_pipeline import DataflowPipeline
from SystemX.pipeline.dataflow_pipeline_factory import DataflowPipelineFactory

from ..policy.WorkspacePolicy import WorkspacePolicy
from .SystemXBaseHandler import SystemXBaseHandler

class LabelingGenerateHandler(SystemXBaseHandler):
    @property
    def response_key(self) -> str:
        return "dataflow"

    def create_pipeline(self, input_data: dict) -> DataflowPipeline:
        return DataflowPipelineFactory.create(input_data)

    @tornado.web.authenticated
    def post(self) -> None:
        input_data: dict = self.get_json_body()
        requested_filename: str = input_data.get("filename", "")
        base_path: str = input_data.get("base_path", "")

        cells_payload: list[dict] = input_data.get("cells", [])

        self.log.info(f"Requested filename: {requested_filename}. Base path: {base_path}.")

        if requested_filename and base_path:
            try:
                workspace = WorkspacePolicy(Path.cwd(), base_path)

                json_filename: str = f"{Path(requested_filename).stem}.json"
                target_path: Path = workspace.get_secure_annotation_path(json_filename)

                self.log.info(f"Checking cache for {target_path}")
                if target_path.exists():
                    self.log.info(f"CACHE HIT: Loading JSON annotation from {target_path}")
                    with open(target_path, encoding="utf-8") as f:
                        cached_payload = json.load(f)

                    if isinstance(cached_payload, dict):
                        cached_elements: list[dict] = cached_payload.get("elements", [])
                    elif isinstance(cached_payload, list):
                        cached_elements: list[dict] = cached_payload
                    else:
                        self.log.error(f"Unexpected JSON schema in {target_path}. Expected dict or list.")
                        cached_elements = []

                    repaired_elements = self._repair_fallback_ids(cached_elements, cells_payload)

                    self.finish(
                        json.dumps(
                            {
                                "message": "Loaded existing annotations from disk.",
                                "success": True,
                                self.response_key: {"elements": repaired_elements},
                            }
                        )
                    )
                    return

            except (ValueError, PermissionError) as e:
                self.log.warning(f"Security validation failed during cache check: {e}")
            except Exception as e:
                self.log.error(f"Failed to load JSON: {e}. Falling back to generation.", exc_info=True)

        self.log.info("CACHE MISS: Executing raw generation pipeline.")
        super().post()

    def _repair_fallback_ids(self, elements: list[dict], cells: list[dict]) -> list[dict]:
        """Map 'fallback-X' cell_ids to the actual Jupyter cell IDs from the frontend."""
        if not cells:
            return elements

        ordered_actual_ids = [cell.get("id") for cell in cells if cell.get("id")]

        for element in elements:
            data = element.get("data", {})
            current_id = data.get("cell_id", "")

            if isinstance(current_id, str) and current_id.startswith("fallback-"):
                try:
                    idx = int(current_id.split("-")[1]) - 1
                    if 0 <= idx < len(ordered_actual_ids):
                        data["cell_id"] = ordered_actual_ids[idx]
                except (IndexError, ValueError):
                    self.log.warning(f"Failed to map {current_id} to an actual cell ID.")

        return elements
