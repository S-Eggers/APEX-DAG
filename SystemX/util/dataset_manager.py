import json
import logging
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

class FlagType(StrEnum):
    """Domain definition for supported notebook flags."""

    BUG_IN_DATAFLOW = "bug_in_dataflow"
    NEEDS_REVIEW = "needs_review"
    NOT_RELEVANT = "not_relevant"

class DatasetManager:
    """Stateless manager leveraging set operations and positional offsets."""

    @classmethod
    def get_next_unannotated(cls, raw_dir: Path, annotations_dir: Path, flags_registry: Path, current_filename: str | None = None) -> str | None:

        raw_files = cls._get_raw_files(raw_dir)
        if not raw_files:
            return None

        annotated_files = cls._get_annotated_files(annotations_dir)
        flagged_files = set(cls._load_flags(flags_registry).keys())

        unannotated = list(raw_files - annotated_files - flagged_files)
        return cls._get_next_in_sequence(unannotated, current_filename)

    @classmethod
    def get_next_annotated(cls, annotations_dir: Path, current_filename: str | None = None) -> str | None:
        """Reverse mode: Targets only notebooks that have been annotated."""
        logger.info("Fetching next annotated notebook.")
        annotated_files = list(cls._get_annotated_files(annotations_dir))
        return cls._get_next_in_sequence(annotated_files, current_filename)

    @classmethod
    def get_next_flagged(cls, flags_registry: Path, target_flag: FlagType, current_filename: str | None = None) -> str | None:
        """Targets only notebooks marked with a specific flag type."""
        all_flags = cls._load_flags(flags_registry)

        flagged_files = [filename for filename, flag_val in all_flags.items() if flag_val == target_flag.value]
        return cls._get_next_in_sequence(flagged_files, current_filename)

    @staticmethod
    def _get_next_in_sequence(items: list[str], current_item: str | None) -> str | None:
        """Handles positional offsets strictly separated from I/O logic."""
        if not items:
            return None

        items.sort()

        if current_item and current_item in items:
            current_idx = items.index(current_item)
            return items[(current_idx + 1) % len(items)]

        return items[0]

    @staticmethod
    def _get_raw_files(raw_dir: Path) -> set[str]:
        if not raw_dir.exists():
            logger.error(f"Dataset directory not found: {raw_dir}")
            return set()
        return {f.name for f in raw_dir.glob("*.ipynb")}

    @staticmethod
    def _get_annotated_files(annotations_dir: Path) -> set[str]:
        if not annotations_dir.exists():
            return set()
        return {f.with_suffix(".ipynb").name for f in annotations_dir.glob("*.json")}

    @staticmethod
    def _load_flags(flags_registry: Path) -> dict[str, str]:
        """Parses the JSON registry."""
        if not flags_registry.exists():
            return {}

        try:
            with open(flags_registry, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Malformed JSON in flags registry {flags_registry}: {e}")
            return {}
        except OSError as e:
            logger.error(f"I/O error reading flags registry {flags_registry}: {e}")
            return {}
