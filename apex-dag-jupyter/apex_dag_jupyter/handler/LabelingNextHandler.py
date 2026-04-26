import json
from pathlib import Path

import tornado
from ApexDAG.util.dataset_manager import DatasetManager
from jupyter_server.base.handlers import APIHandler


class LabelingNextHandler(APIHandler):
    @tornado.web.authenticated
    def post(self) -> None:
        try:
            input_data = self.get_json_body()

            requested_raw_path = input_data.get("datasetPath", "raw_dataset")

            workspace_dir = Path.cwd()
            raw_dir = (workspace_dir / requested_raw_path).resolve()

            if not raw_dir.is_relative_to(workspace_dir):
                self.set_status(403)
                self.finish(
                    json.dumps({"success": False, "message": "Path traversal blocked."})
                )
                return

            annotations_dir = Path.home() / ".apexdag" / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)

            next_filename = DatasetManager.get_next_unannotated(
                raw_dir, annotations_dir
            )

            if not next_filename:
                self.finish(
                    json.dumps(
                        {
                            "success": False,
                            "message": f"No more unannotated notebooks found in {raw_dir.name}.",
                        }
                    )
                )
                return

            relative_path = f"{requested_raw_path}/{next_filename}"
            relative_path = relative_path.replace("//", "/")

            self.finish(
                json.dumps(
                    {"success": True, "path": relative_path, "filename": next_filename}
                )
            )

        except Exception as e:
            self.log.error(f"Error fetching next notebook: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))
