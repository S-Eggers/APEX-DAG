import json
from pathlib import Path
from typing import TypedDict

import tornado
from SystemX.util.dataset_manager import FlagType
from jupyter_server.base.handlers import APIHandler

from ..policy.FlagRepository import FlagRepository
from ..policy.WorkspacePolicy import WorkspacePolicy


class FlagRequestPayload(TypedDict, total=False):
    filename: str
    reason: str
    base_path: str


class LabelingFlagHandler(APIHandler):
    @tornado.web.authenticated
    def post(self) -> None:
        try:
            input_data: FlagRequestPayload = self.get_json_body()
            filename: str | None = input_data.get("filename")
            reason_str: str | None = input_data.get("reason")
            base_path: str | None = input_data.get("base_path")

            if not filename or not reason_str or not base_path:
                self.set_status(400)
                self.finish(json.dumps({"success": False, "message": "Missing filename, reason, or base_path."}))
                return

            try:
                flag_reason = FlagType(reason_str)
            except ValueError:
                self.set_status(400)
                self.finish(json.dumps({"success": False, "message": f"Invalid flag reason: {reason_str}"}))
                return

            workspace = WorkspacePolicy(Path.cwd(), base_path)
            workspace.ensure_directories()

            flags_registry: Path = workspace.get_flags_registry_path()
            flag_repo = FlagRepository(flags_registry)
            flag_repo.register_flag(filename, flag_reason)

            self.finish(json.dumps({"success": True, "message": f"Flagged {filename} as {flag_reason.value}"}))

        except (ValueError, PermissionError) as e:
            self.set_status(403)
            self.finish(json.dumps({"success": False, "message": str(e)}))
        except Exception as e:
            self.log.error(f"Flag error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Server Error"}))
