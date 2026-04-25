import json
import tornado
from pathlib import Path
from jupyter_server.base.handlers import APIHandler
from ApexDAG.util.dataset_manager import DatasetManager

class LabelingNextHandler(APIHandler):
    @tornado.web.authenticated
    def get(self):
        try:
            workspace_dir = Path.cwd() 
            raw_dir = workspace_dir / "jetbrains_dfg_100k_new" / "code"
            
            annotations_dir = Path.home() / ".apexdag" / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)

            next_filename = DatasetManager.get_next_unannotated(raw_dir, annotations_dir)

            if not next_filename:
                self.finish(json.dumps({
                    "success": False, 
                    "message": "No more unannotated notebooks found in the dataset directory."
                }))
                return

            relative_path = f"jetbrains_dfg_100k_new/code/{next_filename}"

            self.finish(json.dumps({
                "success": True, 
                "path": relative_path,
                "filename": next_filename
            }))

        except Exception as e:
            self.log.error(f"Error fetching next notebook: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))