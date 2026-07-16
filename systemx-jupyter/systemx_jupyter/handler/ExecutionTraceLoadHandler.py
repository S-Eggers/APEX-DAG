import json
from pathlib import Path
from typing import Any

import tornado.web
from jupyter_server.base.handlers import APIHandler

from ..policy.WorkspacePolicy import WorkspacePolicy
from .ExecutionTraceSaveHandler import trace_storage_name


class ExecutionTraceLoadHandler(APIHandler):
    @tornado.web.authenticated
    def post(self) -> None:
        try:
            input_data: dict[str, Any] = self.get_json_body()
            filename: str = input_data.get("filename", "")
            base_path: str = input_data.get("base_path", "")

            if not base_path or not filename:
                self.set_status(400)
                self.finish(json.dumps({"success": False, "message": "Missing base_path or filename."}))
                return

            workspace = WorkspacePolicy(Path.cwd(), base_path)
            target_path = workspace.get_secure_trace_path(trace_storage_name(filename))

            trace = None
            if target_path.exists():
                try:
                    trace = json.loads(target_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    self.log.warning("Unreadable trace sidecar at %s.", target_path)

            self.finish(json.dumps({"success": True, "trace": trace}))

        except (ValueError, PermissionError) as e:
            self.set_status(403)
            self.finish(json.dumps({"success": False, "message": str(e)}))
        except Exception as e:
            self.log.error(f"Trace load error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))
