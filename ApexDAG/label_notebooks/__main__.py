import argparse
import json
import logging
import random
import time
from pathlib import Path

import nbformat
from dotenv import load_dotenv
from tqdm import tqdm

from ApexDAG.labeler.llm_labeler import LLMLabeler
from ApexDAG.llm.gemini_provider import GeminiProvider
from ApexDAG.llm.models import load_config
from ApexDAG.pipeline.labeling_pipeline import LabelingPipeline
from ApexDAG.pipeline.labeling_pipeline_factory import LabelingPipelineFactory
from ApexDAG.util.logger import configure_apexdag_logger

from .llm_policy import ExecutionPolicy
from .models import NotebookCellData

configure_apexdag_logger()
logger = logging.getLogger(__name__)


def extract_structured_code(ipynb_path: Path) -> list[NotebookCellData]:
    try:
        with open(ipynb_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        cells: list[NotebookCellData] = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                cell_id = getattr(cell, "id", f"fallback-{i}")
                cells.append({"cell_id": cell_id, "source": cell.source})
        return cells
    except Exception as e:
        logger.error(f"Failed to parse notebook {ipynb_path.name}: {e}")
        return []


def execute_with_backoff(pipeline: LabelingPipeline, cells: list[NotebookCellData], max_retries: int = 5) -> dict:
    """
    Wraps the pipeline execution with exponential backoff and jitter.
    Targets 429 (Rate Limit) and 503 (Server Busy) scenarios.
    """
    initial_delay = 30.0
    factor = 2.0

    for attempt in range(max_retries):
        try:
            return pipeline.execute(cells)
        except Exception as e:
            err_msg = str(e).lower()
            is_retryable = any(x in err_msg for x in ["429", "quota", "exhausted", "503", "busy", "limit"])

            if not is_retryable or attempt == max_retries - 1:
                raise e

            # Exponential backoff with "Full Jitter"
            delay = initial_delay * (factor**attempt)
            jittered_delay = random.uniform(1.0, delay)

            logger.warning(f"Transient error detected: {e}. Retrying in {jittered_delay:.2f}s (Attempt {attempt + 1}/{max_retries})")
            time.sleep(jittered_delay)

    return {}


def get_processing_queue(input_dir: Path, output_dir: Path) -> list[Path]:
    all_notebooks = list(input_dir.glob("*.ipynb"))
    processed_stems = {f.stem for f in output_dir.glob("*.json")}
    return [f for f in all_notebooks if f.stem not in processed_stems]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="ApexDAG Batch Labeling Engine")
    parser.add_argument("--config", type=str, default="ApexDAG/label_notebooks/config.yaml")
    parser.add_argument("--input_dir", type=str, default="v2_labeling_workspace/raw_dataset")
    parser.add_argument("--output_dir", type=str, default="v2_labeling_workspace/auto_labelled")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input_dir)

    provider = GeminiProvider(model_name=config.model_name)
    policy = ExecutionPolicy(max_tokens=config.max_tokens, max_rpm=config.max_rpm)

    request_payload = {"llmClassification": getattr(config, "use_llm", True), "useRefiner": getattr(config, "use_refiner", True), "replaceDataflowInUDFs": getattr(config, "replace_dataflow", False)}

    pipeline = LabelingPipelineFactory.create(request_payload, model={})
    if isinstance(pipeline.labeler, LLMLabeler):
        pipeline.labeler.configure(config, provider, policy)

    pending_files = get_processing_queue(input_path, output_path)
    logger.info(f"Starting batch: {len(pending_files)} pending notebooks.")

    for ipynb_file in tqdm(pending_files, desc="Batch Processing"):
        if policy.stop_event.is_set():
            logger.error("Resource constraints reached. Terminating.")
            break

        try:
            structured_cells = extract_structured_code(ipynb_file)
            if not structured_cells:
                continue

            result_dict = execute_with_backoff(pipeline, structured_cells)

            output_file = output_path / f"{ipynb_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Critical failure on {ipynb_file.name}: {e}", exc_info=True)

    logger.info(f"Process complete. Total tokens: {policy.total_used}")


if __name__ == "__main__":
    main()
