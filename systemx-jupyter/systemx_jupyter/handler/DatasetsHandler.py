import json
import os
from pathlib import Path
from typing import Any

import tornado
from jupyter_server.base.handlers import APIHandler

_RESERVED_NAMES = {"notebooks", "annotations", "annotations_backup", "errors"}

_MAX_DEPTH = 3

def _count(directory: Path, suffix: str) -> int:
    """Count files with the given suffix directly in a directory."""
    try:
        with os.scandir(directory) as entries:
            return sum(1 for entry in entries if entry.name.endswith(suffix) and entry.is_file())
    except OSError:
        return 0

def _walk(directory: Path, depth: int) -> "list[Path]":
    """Yield subdirectories up to the maximum depth, skipping hidden and reserved names."""
    if depth > _MAX_DEPTH:
        return
    try:
        children = sorted(child for child in directory.iterdir() if child.is_dir())
    except OSError:
        return
    for child in children:
        if child.name.startswith(".") or child.name in _RESERVED_NAMES:
            continue
        yield child
        yield from _walk(child, depth + 1)

def discover_datasets(data_dir: Path, root: Path) -> list[dict[str, Any]]:
    """Return descriptors for every directory beneath data_dir that contains a notebooks/ subdirectory."""
    found: list[dict[str, Any]] = []
    for candidate in _walk(data_dir, depth=1):
        notebooks_dir = candidate / "notebooks"
        if not notebooks_dir.is_dir():
            continue

        annotations_dir = candidate / "annotations"
        has_annotations = annotations_dir.is_dir()
        found.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "label": candidate.relative_to(data_dir).as_posix(),
                "notebooks": _count(notebooks_dir, ".ipynb"),
                "annotations": _count(annotations_dir, ".json") if has_annotations else 0,
                "has_annotations": has_annotations,
            }
        )

    found.sort(key=lambda entry: entry["path"])
    return found

class DatasetsHandler(APIHandler):
    """Auto-discovers valid labeling datasets under ./data."""

    @tornado.web.authenticated
    def get(self) -> None:
        try:
            root = Path.cwd().resolve()
            data_dir = (root / "data").resolve()
            datasets = discover_datasets(data_dir, root) if data_dir.is_dir() else []
            self.finish(json.dumps({"success": True, "root": str(root), "datasets": datasets}))
        except Exception as e:
            self.log.error(f"Dataset discovery failed: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Server Error", "datasets": []}))

    def data_received(self, chunk: bytes) -> None:
        pass
