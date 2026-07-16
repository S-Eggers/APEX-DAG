import json
import logging
from pathlib import Path

from SystemX.util.dataset_manager import FlagType

logger = logging.getLogger(__name__)


class FlagRepository:
    """Handles the persistence of notebook flags."""

    def __init__(self, registry_path: Path) -> None:
        self._registry_path: Path = registry_path

    def register_flag(self, filename: str, flag: FlagType) -> None:
        """Saves a flag for a specific notebook."""
        flags_data = self._load()
        flags_data[filename] = flag.value
        self._save(flags_data)

    def _load(self) -> dict[str, str]:
        if not self._registry_path.exists():
            return {}
        try:
            with open(self._registry_path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted flags registry at {self._registry_path}. Resetting. Error: {e}")
            return {}

    def _save(self, data: dict[str, str]) -> None:
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
