import json
import networkx as nx
from pathlib import Path
import tornado
from jupyter_server.base.handlers import APIHandler

class LabelingSaveHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        try:
            input_data = self.get_json_body()
            requested_filename = input_data.get("filename", "annotated_graph.gml")
            elements = input_data.get("graph", [])
            code_data = input_data.get("code", "")

            safe_base_dir = Path.home() / ".apexdag" / "annotations"
            safe_base_dir.mkdir(parents=True, exist_ok=True)

            secure_filename = Path(requested_filename).name
            target_path = (safe_base_dir / secure_filename).resolve()

            if not target_path.is_relative_to(safe_base_dir):
                self.log.error(f"Path traversal blocked: {requested_filename}")
                self.set_status(403)
                self.finish(json.dumps({"success": False, "message": "Forbidden path."}))
                return

            if target_path.suffix != '.gml':
                target_path = target_path.with_suffix('.gml')

            raw_elements = input_data.get("graph", [])
            code_data = input_data.get("code", "")

            if isinstance(raw_elements, dict):
                elements = raw_elements.get("nodes", []) + raw_elements.get("edges", [])
            else:
                elements = raw_elements

            G = nx.DiGraph()

            for el in elements:
                data = el.get("data", {})
                clean_data = {}
                for k, v in data.items():
                    if v is None:
                        continue
                    if isinstance(v, (dict, list)):
                        clean_data[k] = json.dumps(v)
                    else:
                        clean_data[k] = v

                if "source" in clean_data and "target" in clean_data:
                    u = clean_data.pop("source")
                    v = clean_data.pop("target")
                    clean_data.pop("id", None)
                    G.add_edge(u, v, **clean_data)
                elif "id" in clean_data:
                    n = clean_data.pop("id")
                    G.add_node(n, **clean_data)

            nx.write_gml(G, target_path)

            code_path = target_path.with_suffix('.py')
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code_data)

            self.finish(json.dumps({
                "success": True, 
                "message": f"Saved GML to {target_path.name} and source to {code_path.name}"
            }))

        except Exception as e:
            self.log.error(f"Save error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))