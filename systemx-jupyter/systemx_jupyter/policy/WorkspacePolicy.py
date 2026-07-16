from pathlib import Path


class WorkspacePolicy:
    """Resolves workspace directories while preventing path traversal outside the root."""

    def __init__(self, root_dir: Path, requested_base: str) -> None:
        self._root_dir: Path = root_dir.resolve()

        if not requested_base:
            raise ValueError("base_path must be provided.")

        safe_relative_base: str = requested_base.lstrip("/\\")
        self.base_dir: Path = (self._root_dir / safe_relative_base).resolve()

        if not self.base_dir.is_relative_to(self._root_dir):
            raise PermissionError("Path traversal attempt detected. Access denied.")

        self.notebooks_dir: Path = self.base_dir / "notebooks"
        self.annotations_dir: Path = self.base_dir / "annotations"
        self.traces_dir: Path = self.base_dir / "traces"

    def ensure_directories(self) -> None:
        """Ensure the strictly required directory structure exists."""
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def get_secure_annotation_path(self, filename: str) -> Path:
        """Sanitizes the filename and returns an absolute path inside annotations_dir."""
        secure_filename: str = Path(filename).name
        return (self.annotations_dir / secure_filename).resolve()

    def get_secure_trace_path(self, filename: str) -> Path:
        """Sanitizes the filename and returns an absolute path inside traces_dir."""
        secure_filename: str = Path(filename).name
        return (self.traces_dir / secure_filename).resolve()

    def get_flags_registry_path(self) -> Path:
        """Returns the localized registry path for this specific workspace."""
        return (self.base_dir / "notebook_flags.json").resolve()
