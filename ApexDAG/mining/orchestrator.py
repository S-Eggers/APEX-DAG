import json
import logging
from pathlib import Path

from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Stateless manager leveraging set operations for instant unannotated
    file resolution.
    """

    @classmethod
    def get_next_unannotated(cls, raw_dir: Path, annotations_dir: Path) -> str | None:
        if not raw_dir.exists():
            logger.error(f"Dataset directory not found: {raw_dir}")
            return None

        # 1. Gather all raw notebook filenames
        raw_files = {f.name for f in raw_dir.glob("*.ipynb")}

        # 2. Gather completed annotations
        annotated_files = {
            f.name.replace(".json", ".ipynb") for f in annotations_dir.glob("*.json")
        }

        # 3. Gather flagged notebooks from the registry
        flagged_files = set()
        flags_registry = annotations_dir.parent / "notebook_flags.json"

        if flags_registry.exists():
            try:
                with open(flags_registry, encoding="utf-8") as f:
                    flags_data = json.load(f)
                    flagged_files = {name for name in flags_data}
            except Exception as e:
                logger.error(f"Failed to read flags registry: {e}")

        # 4. Pure stateless diff
        unannotated = list(raw_files - annotated_files - flagged_files)

        if not unannotated:
            return None

        unannotated.sort()
        return unannotated[0]
