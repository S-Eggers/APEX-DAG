import logging
from pathlib import Path

import nbformat

from ApexDAG.util.logger import configure_apexdag_logger

from .models import NotebookCellData

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class NotebookExtractor:
    @staticmethod
    def to_structured_cells(path: Path, greedy: bool = True) -> list[NotebookCellData]:
        try:
            with open(path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            raw_cells = []
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue

                source = cell.source.strip()
                if not source:
                    continue

                exec_count = cell.get("execution_count")
                cell_id = getattr(cell, "id", f"fallback-{i}")

                raw_cells.append({"cell_id": str(cell_id), "source": cell.source, "execution_count": exec_count})

            if not greedy:
                executed_cells = [c for c in raw_cells if c["execution_count"] is not None and c["execution_count"] > 0]
                executed_cells.sort(key=lambda x: x["execution_count"])
                return [{"cell_id": c["cell_id"], "source": c["source"]} for c in executed_cells]

            return [{"cell_id": c["cell_id"], "source": c["source"]} for c in raw_cells]

        except Exception as e:
            logger.error(f"Critical failure parsing {path}: {e}")
            return []
