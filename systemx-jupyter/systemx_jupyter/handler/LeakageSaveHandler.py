import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tornado.web
from jupyter_server.base.handlers import APIHandler

from ..policy.WorkspacePolicy import WorkspacePolicy

logger = logging.getLogger(__name__)


def leakage_sidecar_name(notebook_name: str) -> str:
    """Return the leakage gold sidecar filename for a notebook."""
    return f"{Path(notebook_name).stem}.leakage.json"


def extract_gold(raw_graph: dict[str, Any] | list[Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Extract the per-node human gold and detected leakage classes from a Cytoscape graph."""
    if isinstance(raw_graph, dict):
        elements = raw_graph.get("nodes", []) + raw_graph.get("edges", [])
    elif isinstance(raw_graph, list):
        elements = raw_graph
    else:
        raise ValueError("Invalid graph payload; expected list or dict.")

    gold: dict[str, str] = {}
    detected: dict[str, str] = {}
    for element in elements:
        data = element.get("data") if isinstance(element, dict) else None
        if not isinstance(data, dict):
            continue
        node_id = data.get("id")
        if not node_id:
            continue
        if data.get("leakage_gold"):
            gold[str(node_id)] = str(data["leakage_gold"])
        if data.get("leakage_class"):
            detected[str(node_id)] = str(data["leakage_class"])
    return gold, detected


class LeakageSaveHandler(APIHandler):
    """Persists the human's per-node leakage gold labels as a sidecar file."""

    @tornado.web.authenticated
    def post(self) -> None:
        try:
            input_data: dict[str, Any] = self.get_json_body()
            requested_filename: str = input_data.get("filename", "notebook")
            base_path: str = input_data.get("base_path", "")

            if not base_path:
                self.set_status(400)
                self.finish(json.dumps({"success": False, "message": "Missing base_path."}))
                return

            workspace = WorkspacePolicy(Path.cwd(), base_path)
            workspace.ensure_directories()
            target_path: Path = workspace.get_secure_annotation_path(leakage_sidecar_name(requested_filename))

            gold, detected = extract_gold(input_data.get("graph", []))

            payload = {
                "notebook": Path(requested_filename).name,
                "saved_at": datetime.now(UTC).isoformat(),
                "gold": gold,
                "detected": detected,
            }
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.finish(
                json.dumps(
                    {
                        "success": True,
                        "message": f"Saved {len(gold)} leakage label(s) to {target_path.name}",
                    }
                )
            )

        except (ValueError, PermissionError) as e:
            self.set_status(403)
            self.finish(json.dumps({"success": False, "message": str(e)}))
        except Exception as e:
            self.log.error(f"Leakage save error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))
