import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ApexDAG.label_notebooks.llm_policy import ExecutionPolicy
from ApexDAG.label_notebooks.utils import load_config
from ApexDAG.labeler.llm_labeler import LLMLabeler
from ApexDAG.llm.gemini_provider import GeminiProvider
from ApexDAG.pipeline.pipeline_factory import LabelingPipelineFactory
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


def get_processing_queue(input_dir: Path, output_dir: Path) -> list[Path]:
    """Identifies pending notebooks by comparing input and output directories."""
    all_notebooks = list(input_dir.glob("*.ipynb"))
    processed_stems = {f.stem for f in output_dir.glob("*.json")}

    pending = [f for f in all_notebooks if f.stem not in processed_stems]
    return pending


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
            logger.error("Resource constraints reached (Tokens/RPM). Terminating.")
            break

        try:
            with open(ipynb_file, encoding="utf-8") as f:
                nb = json.load(f)
                code_cells = ["".join(cell.get("source", [])) for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
                raw_code = "\n\n".join(code_cells)

            result_dict = pipeline.execute(raw_code)

            # Save Output
            output_file = output_path / f"{ipynb_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Critical failure on {ipynb_file.name}: {e}", exc_info=True)

    logger.info(f"Process complete. Total tokens: {policy.total_used}")


if __name__ == "__main__":
    main()
