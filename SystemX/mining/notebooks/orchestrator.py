import json
import logging
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from SystemX.mining.notebooks.policy import GraphSamplingPolicy
from SystemX.mining.notebooks.validator import PipelineValidator
from SystemX.mining.protocols import NotebookIterator

from .models import ValidationMetrics

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", module="nbformat")

logger = logging.getLogger(__name__)

class MiningOrchestrator:
    """Stateful orchestrator that safely mines and curates Jupyter notebooks."""

    def __init__(
        self,
        target_count: int,
        iterator: NotebookIterator,
        workspace_dir: str | Path,
        target_libraries: set[str] | None = None,
    ) -> None:
        load_dotenv()
        self.target_count = target_count
        self.iterator = iterator
        self.target_libraries = target_libraries or set()

        self.workspace = Path(workspace_dir)
        self.notebooks_dir = self.workspace / "notebooks"
        self.errors_dir = self.workspace / "errors"

        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.workspace / "mining_state.json"
        self.ledger_file = self.workspace / "mining_ledger.jsonl"

        self.validator = PipelineValidator(target_libraries=self.target_libraries)
        self._load_state()

        self.iterator.current_index = self.state["current_index"]

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    self.state: dict[str, int] = json.load(f)

                logger.info(f"Resuming from index {self.state['current_index']}. Currently kept: {self.state['kept_count']}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load state: {e}. Starting fresh.")
                self.state = {"current_index": 0, "kept_count": 0, "processed_count": 0}
        else:
            self.state = {"current_index": 0, "kept_count": 0, "processed_count": 0}

    def _save_state(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=4)

    def _append_to_ledger(self, source_id: str, metrics: ValidationMetrics, kept: bool, reason: str) -> None:
        """Streams metrics and decision reasoning to a JSONL file."""
        record: dict[str, Any] = {
            "source_id": source_id,
            "success": metrics.success,
            "kept": kept,
            "status_reason": reason,
            "edge_count": metrics.edge_count,
            "node_count": metrics.node_count,
            "loc": metrics.lines_of_code,
            "density": round(metrics.edge_count / max(metrics.lines_of_code, 1), 4),
            "ml_semantics": metrics.contains_ml_semantics,
            "matched_libraries": sorted(metrics.matched_libraries),
            "extraction_time": round(metrics.extraction_time_sec, 4),
        }
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def mine(self) -> None:
        logger.info(f"Starting mining. Target: {self.target_count} notebooks.")

        with tqdm(
            total=self.target_count,
            initial=self.state["kept_count"],
            desc="Notebooks Kept",
        ) as pbar:
            for source_id, notebook_dict in self.iterator:
                if self.state["kept_count"] >= self.target_count:
                    logger.info("Target dataset size reached. Terminating.")
                    break

                safe_filename = source_id.split("/")[-1]
                file_stem = safe_filename.replace(".ipynb", "")

                metrics = self.validator.validate(notebook_dict)
                keep, reason = GraphSamplingPolicy.evaluate(
                    metrics, safe_filename, require_libraries=bool(self.target_libraries)
                )

                if keep:
                    nb_path = self.notebooks_dir / safe_filename
                    with open(nb_path, "w", encoding="utf-8") as f:
                        json.dump(notebook_dict, f, indent=2)

                    self.state["kept_count"] += 1
                    pbar.update(1)

                elif not metrics.success and metrics.stacktrace:
                    error_path = self.errors_dir / f"{file_stem}.trace"
                    with open(error_path, "w", encoding="utf-8") as f:
                        f.write(metrics.stacktrace)

                self.state["processed_count"] += 1
                self.state["current_index"] = self.iterator.current_index

                self._append_to_ledger(source_id, metrics, keep, reason)

                if self.state["processed_count"] % 50 == 0:
                    self._save_state()

        self._save_state()
        logger.info(f"Mining complete. Kept {self.state['kept_count']} notebooks.")

    def save_checkpoint(self) -> None:
        """Public method to allow external graceful shutdowns to trigger a state save."""
        self._save_state()
