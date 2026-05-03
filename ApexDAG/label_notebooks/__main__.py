import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from ApexDAG.labeler.llm_labeler import LLMLabeler
from ApexDAG.llm.gemini_provider import GeminiProvider
from ApexDAG.llm.models import load_config
from ApexDAG.llm.resilent_provider import ResilientProvider
from ApexDAG.pipeline.labeling_pipeline_factory import LabelingPipelineFactory
from ApexDAG.util.logger import configure_apexdag_logger

from .engine import BatchLabelingEngine
from .llm_policy import ExecutionPolicy

configure_apexdag_logger()
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="ApexDAG Batch Labeling Engine")
    parser.add_argument("--config", type=str, default="ApexDAG/label_notebooks/config.yaml")
    parser.add_argument("--input_dir", type=str, default="./data/jetbrains_dataset/notebooks")
    parser.add_argument("--output_dir", type=str, default="./data/jetbrains_dataset/annotations")
    args = parser.parse_args()

    config = load_config(args.config)
    base_provider = GeminiProvider(model_name=config.model_name)
    resilient_provider = ResilientProvider(base_provider, max_retries=config.retry_attempts)
    policy = ExecutionPolicy(max_tokens=config.max_tokens, max_rpm=config.max_rpm)

    request_payload = {"llmClassification": getattr(config, "use_llm", True), "useRefiner": getattr(config, "use_refiner", True), "replaceDataflowInUDFs": getattr(config, "replace_dataflow", False)}

    pipeline = LabelingPipelineFactory.create(request_payload, model={})
    if isinstance(pipeline.labeler, LLMLabeler):
        pipeline.labeler.configure(config, resilient_provider, policy)

    engine = BatchLabelingEngine(pipeline=pipeline, output_dir=Path(args.output_dir), policy=policy)

    input_files = list(Path(args.input_dir).glob("*.ipynb"))
    engine.run_batch(input_files)

    logger.info(f"Process complete. Total tokens used: {policy.total_used}")


if __name__ == "__main__":
    main()
