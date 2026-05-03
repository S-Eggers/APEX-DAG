import json
import logging
from pathlib import Path

from tqdm import tqdm

from ApexDAG.pipeline.labeling_pipeline import LabelingPipeline
from ApexDAG.util.logger import configure_apexdag_logger

from .extractor import NotebookExtractor
from .llm_policy import ExecutionPolicy

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class BatchLabelingEngine:
    def __init__(self, pipeline: LabelingPipeline, output_dir: Path, policy: ExecutionPolicy) -> None:
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.policy = policy
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_batch(self, files: list[Path]) -> None:
        # Filter out already successfully processed files
        pending = [f for f in files if not (self.output_dir / f"{f.stem}.json").exists()]
        logger.info(f"Starting batch: {len(pending)} pending notebooks.")

        for ipynb_file in tqdm(pending, desc="Batch Processing"):
            if self.policy.stop_event.is_set():
                logger.error("Token budget reached. Stopping.")
                break

            self._process_file(ipynb_file)

    def _process_file(self, ipynb_file: Path) -> None:
        try:
            cells = NotebookExtractor.to_structured_cells(ipynb_file)
            if not cells:
                return

            result_dict = self.pipeline.execute(cells)

            output_file = self.output_dir / f"{ipynb_file.stem}.json"
            temp_file = output_file.with_suffix(".tmp")

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)

            temp_file.replace(output_file)

        except Exception as e:
            logger.error(f"Critical failure on {ipynb_file.name}: {e}")
