import argparse
import logging
import warnings
from pathlib import Path

from dotenv import load_dotenv

from SystemX.llm.config import load_config
from SystemX.pipeline.llm_labeling_pipeline_factory import LLMLabelingPipelineFactory
from SystemX.util.logger import configure_systemx_logger

from .engine import BatchLabelingEngine

configure_systemx_logger()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=SyntaxWarning, message=".*invalid.*")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="SystemX Batch Labeling Engine")
    parser.add_argument("--config", type=str, default="./SystemX/llm/config/config.yaml")
    parser.add_argument("--input_dir", type=str, default="./data/jetbrains_dataset/notebooks")
    parser.add_argument("--output_dir", type=str, default="./data/jetbrains_dataset/annotations")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = LLMLabelingPipelineFactory.create(config)
    policy = pipeline.labeler.policy
    engine = BatchLabelingEngine(pipeline=pipeline, output_dir=Path(args.output_dir), policy=policy)

    input_files = list(Path(args.input_dir).glob("*.ipynb"))
    engine.run_batch(input_files)

    logger.info(f"Process complete. Total tokens used: {policy.total_used}")


if __name__ == "__main__":
    main()
