import json
from pathlib import Path
from typing import Any

import tornado.web
from SystemX.sca.constants import COMPUTE_HUBS, DOMAIN_EDGES
from jupyter_server.base.handlers import APIHandler

from ..policy.WorkspacePolicy import WorkspacePolicy

CYTOSCAPE_UI_KEYS = {"position", "group", "removed", "selected", "selectable", "locked", "grabbable", "pannable", "classes"}

def sanitize_graph_elements(raw_elements: dict[str, list[dict[str, Any]]] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip Cytoscape UI metadata and resolve domain labels against the taxonomy."""
    elements: list[dict[str, Any]] = []

    if isinstance(raw_elements, dict):
        elements = raw_elements.get("nodes", []) + raw_elements.get("edges", [])
    elif isinstance(raw_elements, list):
        elements = raw_elements
    else:
        raise ValueError("Invalid graph data payload format. Expected list or dict.")

    sanitized_elements: list[dict[str, Any]] = []

    for element in elements:
        clean_element: dict[str, Any] = {key: value for key, value in element.items() if key not in CYTOSCAPE_UI_KEYS}

        data_payload: dict[str, Any] | None = clean_element.get("data")
        if not isinstance(data_payload, dict):
            sanitized_elements.append(clean_element)
            continue

        data_payload.pop("reasoning", None)

        node_type = data_payload.get("node_type")
        predicted_label = data_payload.get("predicted_label")

        if node_type is not None:
            try:
                node_type_int = int(node_type)
            except (ValueError, TypeError):
                sanitized_elements.append(clean_element)
                continue

            is_hub = node_type_int in COMPUTE_HUBS

            if is_hub:
                active_label = int(predicted_label) if predicted_label is not None else -1

                node_meta = DOMAIN_EDGES.get(active_label, {})
                data_payload["domain_label"] = node_meta.get("name", "UNLABELLED")

        clean_element["data"] = data_payload
        sanitized_elements.append(clean_element)

    return sanitized_elements

class LabelingSaveHandler(APIHandler):
    @tornado.web.authenticated
    def post(self) -> None:
        try:
            input_data: dict[str, Any] = self.get_json_body()
            requested_filename: str = input_data.get("filename", "annotated_graph.json")
            base_path: str = input_data.get("base_path", "")

            if not base_path:
                self.set_status(400)
                self.finish(json.dumps({"success": False, "message": "Missing base_path."}))
                return

            workspace = WorkspacePolicy(Path.cwd(), base_path)
            workspace.ensure_directories()

            target_path: Path = workspace.get_secure_annotation_path(requested_filename)

            if target_path.suffix != ".json":
                target_path = target_path.with_suffix(".json")

            raw_elements: dict[str, Any] | list[Any] = input_data.get("graph", [])

            elements = sanitize_graph_elements(raw_elements)

            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(elements, f, indent=2)

            self.finish(json.dumps({"success": True, "message": f"Saved JSON to {target_path.name}"}))

        except (ValueError, PermissionError) as e:
            self.set_status(403)
            self.finish(json.dumps({"success": False, "message": str(e)}))
        except Exception as e:
            self.log.error(f"Save error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))
