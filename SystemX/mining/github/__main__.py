import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from SystemX.mining.github.client import GitHubClient
from SystemX.mining.github.iterator import GitHubNotebookIterator
from SystemX.mining.notebooks.orchestrator import MiningOrchestrator
from SystemX.util.logger import configure_systemx_logger

configure_systemx_logger()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="SystemX GitHub Notebook Mining Orchestrator")
    parser.add_argument("--target", type=int, default=10000, help="Number of notebooks to collect")
    parser.add_argument("--registry", type=str, default="data/github_registry.json", help="Path to the JSON registry of repositories (e.g., result_machine_learning_2024.json)")
    parser.add_argument("--workspace", type=str, default="github_dataset", help="Output directory name for the mined dataset")

    args = parser.parse_args()

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("CRITICAL: GITHUB_TOKEN environment variable is missing. Terminating.")
        sys.exit(1)

    if not os.path.exists(args.registry):
        logger.error(f"CRITICAL: Registry file not found at {args.registry}. Terminating.")
        sys.exit(1)

    base_dir = Path(os.environ.get("RESULTS_DIR", "."))
    workspace_path = base_dir / args.workspace

    try:
        client = GitHubClient(token=github_token)
        iterator = GitHubNotebookIterator(client=client, json_registry_path=args.registry, start_index=0)

        orchestrator = MiningOrchestrator(target_count=args.target, iterator=iterator, workspace_dir=workspace_path)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

    try:
        logger.info(f"Starting GitHub miner targeting {args.target} notebooks via {args.registry}.")
        orchestrator.mine()
    except KeyboardInterrupt:
        logger.info("Mining cleanly interrupted by user. Saving current checkpoint state...")
        orchestrator.save_checkpoint()
    except Exception as e:
        logger.exception(f"Unhandled exception during mining loop: {e}")
        orchestrator.save_checkpoint()
        sys.exit(1)
