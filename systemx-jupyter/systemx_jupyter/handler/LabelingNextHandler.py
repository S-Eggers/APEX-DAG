import json
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict

import tornado
from SystemX.util.dataset_manager import DatasetManager, FlagType
from jupyter_server.base.handlers import APIHandler

from ..policy.WorkspacePolicy import WorkspacePolicy


class FetchMode(StrEnum):
    """Defines the allowed strategies for fetching the next notebook."""

    UNANNOTATED = "unannotated"
    ANNOTATED = "annotated"
    FLAGGED = "flagged"


class NextRequestPayload(TypedDict, total=False):
    """Strict typing for the expected JSON payload."""

    base_path: str
    current_filename: str | None
    mode: str
    flag_type: str | None


class LabelingNextHandler(APIHandler):
    @tornado.web.authenticated
    def post(self) -> None:
        try:
            input_data: NextRequestPayload = self.get_json_body()
            base_path: str | None = input_data.get("base_path")
            current_filename: str | None = input_data.get("current_filename")
            raw_mode: str = input_data.get("mode", FetchMode.UNANNOTATED.value)

            try:
                mode = FetchMode(raw_mode)
            except ValueError:
                self._send_error(400, f"Invalid mode: {raw_mode}. Must be one of {[m.value for m in FetchMode]}")
                return

            if not base_path:
                self._send_error(400, "Missing required field: base_path.")
                return

            workspace = WorkspacePolicy(Path.cwd(), base_path)
            workspace.ensure_directories()
            flags_registry: Path = workspace.get_flags_registry_path()
            self.log.info(f"Fetching next notebook in mode: {mode.value}")

            match mode:
                case FetchMode.UNANNOTATED:
                    next_filename = DatasetManager.get_next_unannotated(workspace.notebooks_dir, workspace.annotations_dir, flags_registry, current_filename)
                case FetchMode.ANNOTATED:
                    next_filename = DatasetManager.get_next_annotated(workspace.annotations_dir, current_filename)
                case FetchMode.FLAGGED:
                    raw_flag = input_data.get("flag_type")
                    if not raw_flag:
                        self._send_error(400, "Missing required field 'flag_type' for FLAGGED mode.")
                        return
                    try:
                        target_flag = FlagType(raw_flag)
                    except ValueError:
                        self._send_error(400, f"Invalid flag_type: {raw_flag}.")
                        return

                    next_filename = DatasetManager.get_next_flagged(flags_registry, target_flag, current_filename)

            if not next_filename:
                self._send_success({"message": f"No target notebooks found in mode: {mode.value}."})
                return

            rel_base: Path = workspace.notebooks_dir.relative_to(Path.cwd())
            relative_path: str = f"{rel_base}/{next_filename}".replace("\\", "/")

            self._send_success({"path": relative_path, "filename": next_filename})

        except (ValueError, PermissionError) as e:
            self._send_error(403, str(e))
        except Exception as e:
            self.log.error(f"Error fetching next notebook: {e}", exc_info=True)
            self._send_error(500, "Internal Server Error")

    def _send_error(self, status_code: int, message: str) -> None:
        self.set_status(status_code)
        self.finish(json.dumps({"success": False, "message": message}))

    def _send_success(self, payload: dict[str, Any]) -> None:
        response = {"success": True}
        response.update(payload)
        self.finish(json.dumps(response))
