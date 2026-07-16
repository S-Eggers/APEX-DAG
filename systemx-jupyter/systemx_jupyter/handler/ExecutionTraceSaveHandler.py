import json
from pathlib import Path
from typing import Any

import tornado.web
from jupyter_server.base.handlers import APIHandler

from ..policy.WorkspacePolicy import WorkspacePolicy

MAX_SOURCE_CHARS = 200_000

_EVENT_KEYS = ("seq", "cell_id", "document_index", "execution_count", "source_hash", "timestamp", "success")
_SESSION_KEYS = ("session_id", "kernel_id", "started_at", "ended_reason")

def trace_storage_name(notebook_path: str) -> str:
    """Map a notebook path to its flattened sidecar trace filename."""
    flattened = notebook_path.replace("\\", "/").strip("/").replace("/", "__")
    return f"{flattened or 'untitled'}.trace.json"

def sanitize_trace(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a whitelisted copy of an incoming trace, dropping unknown keys and malformed entries."""
    if not isinstance(raw, dict):
        raise ValueError("Trace payload must be an object.")

    sessions: list[dict[str, Any]] = []
    for session in raw.get("sessions") or []:
        if not isinstance(session, dict) or not session.get("session_id"):
            continue
        events: list[dict[str, Any]] = []
        for event in session.get("events") or []:
            if not isinstance(event, dict) or "cell_id" not in event:
                continue
            events.append({key: event.get(key) for key in _EVENT_KEYS})
        sessions.append({**{key: session.get(key) for key in _SESSION_KEYS}, "events": events})

    sources = raw.get("sources")
    clean_sources = (
        {str(k): str(v) for k, v in sources.items() if isinstance(v, str) and len(v) <= MAX_SOURCE_CHARS}
        if isinstance(sources, dict)
        else {}
    )

    cells_snapshot = raw.get("cells_snapshot")
    return {
        "schema_version": 1,
        "notebook_path": str(raw.get("notebook_path", "")),
        "saved_at": str(raw.get("saved_at", "")),
        "sessions": sessions,
        "sources": clean_sources,
        "cells_snapshot": cells_snapshot if isinstance(cells_snapshot, list) else [],
    }

def merge_traces(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge an incoming trace into the existing one by session_id, unioning sources."""
    if not existing:
        return incoming

    merged_sessions: list[dict[str, Any]] = []
    incoming_by_id = {s["session_id"]: s for s in incoming["sessions"]}
    for session in existing.get("sessions", []):
        merged_sessions.append(incoming_by_id.pop(session["session_id"], session))
    merged_sessions.extend(incoming_by_id.values())

    sources = dict(existing.get("sources", {}))
    sources.update(incoming["sources"])

    return {**incoming, "sessions": merged_sessions, "sources": sources}

class ExecutionTraceSaveHandler(APIHandler):
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
            workspace.ensure_directories()
            target_path = workspace.get_secure_trace_path(trace_storage_name(filename))

            incoming = sanitize_trace(input_data.get("trace", {}))

            existing: dict[str, Any] | None = None
            if target_path.exists():
                try:
                    existing = json.loads(target_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    self.log.warning("Unreadable existing trace at %s; overwriting.", target_path)

            merged = merge_traces(existing, incoming)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)

            self.finish(json.dumps({"success": True, "message": f"Saved trace to {target_path.name}", "path": str(target_path)}))

        except (ValueError, PermissionError) as e:
            self.set_status(403)
            self.finish(json.dumps({"success": False, "message": str(e)}))
        except Exception as e:
            self.log.error(f"Trace save error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))
