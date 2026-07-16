from pathlib import Path

from SystemX.labeler.llm_labeler import LLMLabeler
from SystemX.labeling.vamsa_loader import (
    DomainEdgeId,
    IOSignatureMappingPolicy,
    VamsaEntry,
    VamsaKBLoader,
)
from SystemX.llm.config import Config
from SystemX.llm.llm_policy import ExecutionPolicy
from SystemX.llm.provider_factory import ProviderFactory
from SystemX.llm.resilient_provider import ResilientProvider
from SystemX.parser.graph_parser import GraphParser
from SystemX.sca.refinement.engine import GraphRefiner
from SystemX.sca.refinement.factory import create_default_refiner, create_empty_refiner

VAMSA_KB_PATH: Path = Path(__file__).parent.parent / "vamsa" / "data" / "knowledge_base.csv"


def _load_vamsa_kb(use_kb: bool = True) -> dict[VamsaEntry, DomainEdgeId]:
    if not use_kb:
        return {}
    return VamsaKBLoader(IOSignatureMappingPolicy()).load_and_map(VAMSA_KB_PATH)


def _build_llm_components(config: Config) -> tuple[GraphParser, LLMLabeler, GraphRefiner]:
    parser = GraphParser(
        replace_dataflow=getattr(config, "replace_dataflow", False),
        enrich_provenance=getattr(config, "enrich_provenance", True),
        detect_dsl=getattr(config, "detect_dsl", False),
    )
    base_provider = ProviderFactory.create(config)
    resilient_provider = ResilientProvider(base_provider, max_retries=config.retry_attempts)
    policy = ExecutionPolicy(max_tokens=config.max_tokens, max_rpm=config.max_rpm)
    kb = _load_vamsa_kb(getattr(config, "use_vamsa_kb", True))
    labeler = LLMLabeler(config=config, provider=resilient_provider, policy=policy, vamsa_mapping=kb)
    refiner = create_default_refiner() if getattr(config, "use_refiner", False) else create_empty_refiner()
    return parser, labeler, refiner
