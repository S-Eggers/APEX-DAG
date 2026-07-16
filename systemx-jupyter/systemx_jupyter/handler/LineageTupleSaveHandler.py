import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tornado.web
from jupyter_server.base.handlers import APIHandler

from ..policy.WorkspacePolicy import WorkspacePolicy

logger = logging.getLogger(__name__)

VALID_TUPLE_TYPES = {"<D, D>", "<M, D>", "<D, Empty>"}

def tuples_sidecar_name(notebook_name: str) -> str:
    """Return the tuple gold sidecar filename for a notebook."""
    return f"{Path(notebook_name).stem}.tuples.json"

def normalize_tuples(raw: Any) -> list[dict[str, str]]:
    """Validate and normalize the incoming tuple list, dropping duplicates."""
    if not isinstance(raw, list):
        raise ValueError("Invalid tuples payload; expected a list.")

    seen: set[tuple[str, str, str]] = set()
    tuples: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tuple_type = str(item.get("tuple_type", ""))
        subject_id = str(item.get("subject_id", ""))
        object_id = str(item.get("object_id", ""))
        if tuple_type not in VALID_TUPLE_TYPES or not subject_id:
            continue
        if tuple_type == "<D, Empty>":
            object_id = "Empty"
        elif not object_id:
            continue
        identity = (tuple_type, subject_id, object_id)
        if identity in seen:
            continue
        seen.add(identity)
        tuples.append(
            {"tuple_type": tuple_type, "subject_id": subject_id, "object_id": object_id}
        )
    return tuples

class LineageTupleSaveHandler(APIHandler):
    """Persists the human's curated lineage-tuple gold set as a sidecar file."""

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
            target_path: Path = workspace.get_secure_annotation_path(
                tuples_sidecar_name(requested_filename)
            )

            tuples = normalize_tuples(input_data.get("tuples", []))

            payload = {
                "notebook": Path(requested_filename).name,
                "saved_at": datetime.now(UTC).isoformat(),
                "tuples": tuples,
            }
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.finish(
                json.dumps(
                    {
                        "success": True,
                        "message": f"Saved {len(tuples)} tuple(s) to {target_path.name}",
                    }
                )
            )

        except (ValueError, PermissionError) as e:
            self.set_status(403)
            self.finish(json.dumps({"success": False, "message": str(e)}))
        except Exception as e:
            self.log.error(f"Tuple save error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))
