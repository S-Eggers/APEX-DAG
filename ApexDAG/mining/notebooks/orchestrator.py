import json
import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ApexDAG.mining.notebooks.iterator import JetbrainsNotebookIterator
from ApexDAG.mining.notebooks.policy import GraphSamplingPolicy
from ApexDAG.mining.notebooks.validator import PipelineValidator, ValidationMetrics
from ApexDAG.util.logger import configure_apexdag_logger

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", module="nbformat")

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class MiningOrchestrator:
    """Stateful orchestrator that safely mines and curates Jupyter notebooks."""

    def __init__(self, target_count: int) -> None:
        load_dotenv()
        self.target_count = target_count
        self.bucket_url = "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"

        # Workspace Setup
        self.workspace = Path(os.getenv("RESULTS_DIR", ".")) / "jetbrains_dataset"
        self.notebooks_dir = self.workspace / "notebooks"
        self.errors_dir = self.workspace / "errors"
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)

        # State & Ledger Files
        self.state_file = self.workspace / "mining_state.json"
        self.ledger_file = self.workspace / "mining_ledger.jsonl"

        self.validator = PipelineValidator()
        self._load_state()

        self.iterator = JetbrainsNotebookIterator(
            json_file="data/ntbs_list.json",
            bucket_url=self.bucket_url,
            start_index=self.state["current_index"],
        )

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self.state = json.load(f)

                logger.info(f"Resuming from index {self.state['current_index']}. Currently kept: {self.state['kept_count']}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load state: {e}. Starting fresh.")
                self.state = {"current_index": 0, "kept_count": 0, "processed_count": 0}
        else:
            self.state = {"current_index": 0, "kept_count": 0, "processed_count": 0}

    def _save_state(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def _append_to_ledger(self, filename: str, metrics: ValidationMetrics, kept: bool, reason: str) -> None:
        """Streams metrics and decision reasoning to a JSONL file."""
        record = {
            "filename": filename,
            "url": f"{self.bucket_url}{filename}",
            "success": metrics.success,
            "kept": kept,
            "status_reason": reason,
            "edge_count": metrics.edge_count,
            "node_count": metrics.node_count,
            "loc": metrics.lines_of_code,
            "density": round(metrics.edge_count / max(metrics.lines_of_code, 1), 4),
            "ml_semantics": metrics.contains_ml_semantics,
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
            for filename, notebook_dict in self.iterator:
                if self.state["kept_count"] >= self.target_count:
                    logger.info("Target dataset size reached. Terminating.")
                    break

                file_stem = filename.replace(".ipynb", "")

                # 1. Pipeline Validation
                metrics = self.validator.validate(notebook_dict)

                # 2. Policy Sampling
                keep, reason = GraphSamplingPolicy.evaluate(metrics, filename)

                # 3. I/O Management
                if keep:
                    nb_path = self.notebooks_dir / filename
                    with open(nb_path, "w", encoding="utf-8") as f:
                        json.dump(notebook_dict, f, indent=2)

                    self.state["kept_count"] += 1
                    pbar.update(1)

                elif not metrics.success and metrics.stacktrace:
                    error_path = self.errors_dir / f"{file_stem}.trace"
                    with open(error_path, "w", encoding="utf-8") as f:
                        f.write(metrics.stacktrace)

                # 4. State & UI Updates
                self.state["processed_count"] += 1
                self.state["current_index"] = self.iterator.current_index

                self._append_to_ledger(filename, metrics, keep, reason)

                # Checkpoint state every 50 files to avoid disk thrashing
                if self.state["processed_count"] % 50 == 0:
                    self._save_state()

        # Final save
        self._save_state()
        logger.info(f"Mining complete. Kept {self.state['kept_count']} notebooks.")
