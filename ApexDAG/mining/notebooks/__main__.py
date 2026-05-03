import argparse
import logging

from ApexDAG.mining.notebooks.orchestrator import MiningOrchestrator
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ApexDAG Notebook Mining Orchestrator")
    parser.add_argument("--target", type=int, default=10000, help="Number of high-signal notebooks to collect")
    args = parser.parse_args()

    orchestrator = MiningOrchestrator(target_count=args.target)
    try:
        orchestrator.mine()
    except KeyboardInterrupt:
        logger.info("Mining interrupted by user. Saving current progress...")
        orchestrator._save_state()
