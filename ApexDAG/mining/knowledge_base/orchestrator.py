import logging

from ApexDAG.util.logger import configure_apexdag_logger
from ApexDAG.vamsa.lineage import KB

from .auditor import KBAuditor
from .profiler import CorpusProfiler
from .synthesizer import BatchSynthesizer

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class KBMinerOrchestrator:
    """Coordinates the mining phases using injected dependencies."""

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

    def run(self, limit: int = 50) -> None:
        # Load Baseline
        baseline_kb = KB(kb_csv_path=self.baseline_csv_path)
        baseline_df = baseline_kb.knowledge_base

        # Phase 1
        self.profiler.build_cache_and_profile(baseline_kb)

        # Baseline Audit
        auditor = KBAuditor(self.profiler.get_cache())
        baseline_coverage = auditor.evaluate(baseline_kb)
        logger.warning(f"Baseline Operation Coverage: {baseline_coverage:.2%}")

        # Phase 2
        top_missing = self.profiler.get_top_missing(limit=limit)
        enhanced_df = self.synthesizer.synthesize(top_missing, baseline_df)

        # Phase 3
        enhanced_kb = KB(knowledge_base=enhanced_df)

        enhanced_coverage = auditor.evaluate(enhanced_kb)
        logger.warning(f"Enhanced Operation Coverage: {enhanced_coverage:.2%}")

        if enhanced_coverage > baseline_coverage:
            logger.warning(f"Success! Coverage increased by {(enhanced_coverage - baseline_coverage):.2%}.")
            self.current_kb = enhanced_kb

            clean_df = enhanced_df.dropna(subset=["API Name"]).copy()

            # 1. Strip the hardcoded fallback rules to prevent geometric explosion
            clean_df = clean_df[~((clean_df["Caller"] == "data") & (clean_df["API Name"].isin(["Subscript", "drop"])))]

            # 2. Strict Deduplication by API signature. We keep 'last' to let newer LLM generations override older ones.
            signature_cols = ["Library", "Module", "Caller", "API Name"]
            clean_df = clean_df.drop_duplicates(subset=signature_cols, keep="last").sort_values(signature_cols)

            clean_df.to_csv(self.output_csv_path, index=False)
            logger.info(f"Saved {len(clean_df)} deduplicated KB entries to {self.output_csv_path}")
