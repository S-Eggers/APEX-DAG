import json
from pathlib import Path
import tornado
from jupyter_server.base.handlers import APIHandler

class LabelingSaveHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        try:
            input_data = self.get_json_body()
            requested_filename = input_data.get("filename", "annotated_graph.json")
            
            safe_base_dir = Path.home() / ".apexdag" / "annotations"
            safe_base_dir.mkdir(parents=True, exist_ok=True)

            secure_filename = Path(requested_filename).name
            target_path = (safe_base_dir / secure_filename).resolve()

            if not target_path.is_relative_to(safe_base_dir):
                self.set_status(403)
                self.finish(json.dumps({"success": False, "message": "Forbidden path."}))
                return

            # Force JSON extension
            if target_path.suffix != '.json':
                target_path = target_path.with_suffix('.json')

            raw_elements = input_data.get("graph", [])
            code_data = input_data.get("code", "")

            # Normalize to flat list (Cytoscape native)
            if isinstance(raw_elements, dict):
                elements = raw_elements.get("nodes", []) + raw_elements.get("edges", [])
            else:
                elements = raw_elements

            # Dump directly to disk
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(elements, f, indent=2)

            code_path = target_path.with_suffix('.py')
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code_data)

            self.finish(json.dumps({
                "success": True, 
                "message": f"Saved JSON to {target_path.name}"
            }))

        except Exception as e:
            self.log.error(f"Save error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))