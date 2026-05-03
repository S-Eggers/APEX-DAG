import argparse
import logging
import os
import sys
import warnings
from collections.abc import Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ApexDAG.llm.gemini_provider import GeminiProvider
from ApexDAG.mining.knowledge_base.orchestrator import KBMinerOrchestrator
from ApexDAG.mining.knowledge_base.profiler import CorpusProfiler
from ApexDAG.mining.knowledge_base.synthesizer import BatchSynthesizer
from ApexDAG.prompts.KBPropagationTemplate import KBPropagationTemplate
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=SyntaxWarning, message=".*invalid.*")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""ApexDAG Knowledge Base Miner: Automatically profile
        and synthesize missing Vamsa API annotations."""
    )

    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to the directory containing raw Python/Jupyter notebooks.",
    )
    parser.add_argument(
        "--baseline-kb",
        type=str,
        required=True,
        help="Path to the current CSV Knowledge Base.",
    )
    parser.add_argument(
        "--output-kb",
        type=str,
        required=True,
        help="Path where the newly enhanced CSV Knowledge Base will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of missing operations to synthesize in this run (default: 50).",
    )

    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        logger.error(
            """GEMINI_API_KEY environment variable is not set.
            Cannot run synthesizer."""
        )
        return 1

    if not os.path.exists(args.corpus):
        logger.error(f"Corpus path does not exist: {args.corpus}")
        return 1

    try:
        profiler = CorpusProfiler(corpus_path=args.corpus)

        gemini_provider = GeminiProvider(model_name="gemini-3.1-flash-lite-preview", api_key=os.getenv("GEMINI_API_KEY"))
        kb_propagation_template = KBPropagationTemplate()
        synthesizer = BatchSynthesizer(gemini_provider, kb_propagation_template)

        orchestrator = KBMinerOrchestrator(
            profiler=profiler,
            synthesizer=synthesizer,
            baseline_csv_path=args.baseline_kb,
            output_csv_path=args.output_kb,
        )

        logger.info("Initializing Continuous Mining Engine...")
        orchestrator.initialize()

        iteration = 1
        while True:
            logger.info(f"\n{'=' * 60}\nStarting Mining Iteration {iteration}\n{'=' * 60}")

            orchestrator.run_iteration(limit=args.limit)

            try:
                user_input = input("\n[?] Do you want to run another synthesis round? (y/n): ").strip().lower()
                if user_input != "y":
                    logger.info("Terminating mining engine. Memory cleared.")
                    break
            except KeyboardInterrupt:
                logger.info("\nProcess aborted by user. Terminating mining engine.")
                break

            iteration += 1

        return 0

    except Exception as e:
        logger.exception(f"KB Miner failed during execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
