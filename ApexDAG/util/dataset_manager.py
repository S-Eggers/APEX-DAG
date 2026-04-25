import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetManager:
    _queue = []
    _iterator = None
    _seen_in_session = set()

    @classmethod
    def get_next_unannotated(cls, raw_dir: Path, annotations_dir: Path) -> str:
        if not raw_dir.exists():
            logger.error(f"Dataset directory not found: {raw_dir}")
            return None

        if cls._iterator is None:
            cls._iterator = os.scandir(raw_dir)

        while len(cls._queue) < 10:
            try:
                entry = next(cls._iterator)
                if not entry.is_file() or not entry.name.endswith('.ipynb'):
                    continue

                if entry.name in cls._seen_in_session:
                    continue

                expected_gml = annotations_dir / f"{entry.name}.gml"
                if not expected_gml.exists():
                    cls._queue.append(entry.name)

            except StopIteration:
                break

        if not cls._queue:
            return None

        next_file = cls._queue.pop(0)
        cls._seen_in_session.add(next_file)
        return next_file
