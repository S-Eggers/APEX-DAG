import json
import logging
from pathlib import Path

from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Stateless manager leveraging set operations and positional
    offsets.
    """

    @classmethod
    def get_next_unannotated(
        cls, raw_dir: Path, annotations_dir: Path, current_filename: str | None = None
    ) -> str | None:

        if not raw_dir.exists():
            logger.error(f"Dataset directory not found: {raw_dir}")
            return None

        raw_files = {f.name for f in raw_dir.glob("*.ipynb")}

        annotated_files = {
            f.name.replace(".json", ".ipynb") for f in annotations_dir.glob("*.json")
        }

        flagged_files = set()
        flags_registry = annotations_dir.parent / "notebook_flags.json"

        if flags_registry.exists():
            try:
                with open(flags_registry, encoding="utf-8") as f:
                    flags_data = json.load(f)
                    flagged_files = {name for name in flags_data}
            except Exception as e:
                logger.error(f"Failed to read flags registry: {e}")

        unannotated = list(raw_files - annotated_files - flagged_files)

        if not unannotated:
            return None

        unannotated.sort()

        if current_filename and current_filename in unannotated:
            current_idx = unannotated.index(current_filename)
            if current_idx + 1 < len(unannotated):
                return unannotated[current_idx + 1]
            else:
                return unannotated[0]

        return unannotated[0]
