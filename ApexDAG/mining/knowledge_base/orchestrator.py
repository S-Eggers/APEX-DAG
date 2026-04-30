import logging

from ApexDAG.util.logger import configure_apexdag_logger
from ApexDAG.vamsa.lineage import KB

from .auditor import KBAuditor
from .profiler import CorpusProfiler
from .synthesizer import BatchSynthesizer

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class KBMinerOrchestrator:
    """Coordinates stateful mining iterations using injected dependencies."""

    def __init__(
        self,
        profiler: CorpusProfiler,
        synthesizer: BatchSynthesizer,
        baseline_csv_path: str,
        output_csv_path: str,
    ) -> None:
        self.profiler = profiler
        self.synthesizer = synthesizer
        self.baseline_csv_path = baseline_csv_path
        self.output_csv_path = output_csv_path

        self.current_kb: KB | None = None
        self.auditor: KBAuditor | None = None

    def initialize(self) -> None:
        """Sets up the initial state and builds the immutable graph cache."""
        self.current_kb = KB(kb_csv_path=self.baseline_csv_path)
        self.profiler.build_cache()
        self.auditor = KBAuditor(self.profiler.get_cache())

    def run_iteration(self, limit: int = 50) -> bool:
        if not self.current_kb or not self.auditor:
            raise RuntimeError("Orchestrator must be initialized before running iterations.")

        self.profiler.profile_missing_operations(self.current_kb)

        baseline_coverage = self.auditor.evaluate(self.current_kb)
        logger.warning(f"Current Operation Coverage: {baseline_coverage:.2%}")

        top_missing = self.profiler.get_top_missing(limit=limit)
        if not top_missing:
            logger.info("No missing operations found! The KB is perfectly complete.")
            return False

        current_df = self.current_kb.knowledge_base
        enhanced_df = self.synthesizer.synthesize(top_missing, current_df)

        clean_df = enhanced_df.dropna(subset=["API Name"]).copy()
        clean_df = clean_df[~((clean_df["Caller"] == "data") & (clean_df["API Name"].isin(["Subscript", "drop"])))]

        signature_cols = ["Library", "Module", "Caller", "API Name"]
        clean_df = clean_df.drop_duplicates(subset=signature_cols, keep="first").sort_values(signature_cols)
        # ---------------------------------------------------------

        enhanced_kb = KB(knowledge_base=clean_df)
        enhanced_coverage = self.auditor.evaluate(enhanced_kb)
        logger.warning(f"Enhanced Operation Coverage: {enhanced_coverage:.2%}")

        if enhanced_coverage > baseline_coverage:
            logger.warning(f"Success! Coverage increased by {(enhanced_coverage - baseline_coverage):.2%}.")
            self.current_kb = enhanced_kb

            clean_df.to_csv(self.output_csv_path, index=False)
            logger.info(f"Saved {len(clean_df)} deduplicated KB entries to {self.output_csv_path}")
            return True
        else:
            logger.error("Synthesis failed to improve coverage. Discarding changes for this iteration.")
            return False
