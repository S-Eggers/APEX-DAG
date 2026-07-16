import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from SystemX.mining.notebooks.iterator import JetbrainsNotebookIterator
from SystemX.mining.notebooks.orchestrator import MiningOrchestrator
from SystemX.util.logger import configure_systemx_logger

configure_systemx_logger()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="SystemX JetBrains Notebook Mining Orchestrator")
    parser.add_argument("--target", type=int, default=10000, help="Number of notebooks to collect")
    parser.add_argument("--registry", type=str, default="data/ntbs_list.json", help="Path to JetBrains JSON registry")
    parser.add_argument("--workspace", type=str, default="jetbrains_dataset", help="Output directory name for the mined dataset")
    args = parser.parse_args()

    JETBRAINS_BUCKET_URL = "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"
    base_dir = Path(os.environ.get("RESULTS_DIR", "."))
    workspace_path = base_dir / args.workspace

    try:
        iterator = JetbrainsNotebookIterator(json_file=args.registry, bucket_url=JETBRAINS_BUCKET_URL, start_index=0)

        orchestrator = MiningOrchestrator(target_count=args.target, iterator=iterator, workspace_dir=workspace_path)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

    try:
        logger.info(f"Starting JetBrains miner targeting {args.target} notebooks. Writing to {workspace_path}")
        orchestrator.mine()
    except KeyboardInterrupt:
        logger.info("Mining cleanly interrupted by user. Saving current checkpoint state...")
        orchestrator.save_checkpoint()
    except Exception as e:
        logger.exception(f"Unhandled exception during mining loop: {e}")
        orchestrator.save_checkpoint()
        sys.exit(1)
