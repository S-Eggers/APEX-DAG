import logging

from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.labeling.labeler import SystemXGraphLabeler
from SystemX.labeling.vamsa_loader import DomainEdgeId, VamsaEntry
from SystemX.llm.config import Config
from SystemX.llm.gemini_provider import GeminiProvider
from SystemX.llm.llm_policy import ExecutionPolicy
from SystemX.llm.llm_provider import StructuredLLMProvider
from SystemX.prompts.NodeClassificationTemplate import NodeClassificationTemplate
from SystemX.sca.constants import DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class LLMLabeler(EdgeLabeler):
    """Stateless edge labeler that delegates execution to SystemXGraphLabeler."""

    def __init__(
        self,
        config: Config | None = None,
        provider: StructuredLLMProvider | None = None,
        policy: ExecutionPolicy | None = None,
        vamsa_mapping: dict[VamsaEntry, DomainEdgeId] | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.policy = policy
        self.vamsa_mapping = vamsa_mapping

    def apply_labels(self, graph: CDFIntermediateRepresentation) -> None:
        if not self.config or not self.provider or not self.policy:
            load_dotenv()
            config = Config(
                model_name="gemini-3.1-flash-lite-preview",
                max_tokens=float("inf"),
                max_rpm=10,
                max_depth=4,
                sleep_interval=0,
                llm_provider="google",
                retry_attempts=2,
                retry_delay=1,
                success_delay=0,
                max_workers=1,
            )
            self.config = config
            self.provider = GeminiProvider(model_name=config.model_name)
            self.policy = ExecutionPolicy(max_tokens=config.max_tokens, max_rpm=config.max_rpm)
            logger.warning("No config injected - falling back to defaults. Token limits and max RPM are reset.")

        logger.info("Starting LLM labeling for graph with %d edges.", len(graph.get_graph().edges))
        labeler = SystemXGraphLabeler(
            config=self.config,
            graph=graph.get_graph(),
            raw_code=graph.get_code(),
            provider=self.provider,
            policy=self.policy,
            template=NodeClassificationTemplate(),
            vamsa_mapping=self.vamsa_mapping,
        )

        batch_size = getattr(self.config, "batch_size", 25)
        labeled_graph, tokens_used = labeler.label_graph(batch_size=batch_size)

        attrs_to_set = _collect_node_labels_as_edge_attrs(labeled_graph)
        graph.set_domain_label(attrs_to_set, name="predicted_label")
        logger.info("Successfully applied labels. Tokens consumed in this session: %d", tokens_used)

def _collect_node_labels_as_edge_attrs(graph: object) -> dict:
    """Reads domain_label from each edge and returns a mapping of edge keys -> predicted_label integer, ready for set_domain_label()."""
    attrs: dict = {}
    for u, v, key, data in graph.edges(data=True, keys=True):
        domain_label = data.get("domain_label", "NOT_RELEVANT").upper()
        attrs[(u, v, key)] = DOMAIN_EDGE_TYPES.get(domain_label, DOMAIN_EDGE_TYPES["NOT_RELEVANT"])
    return attrs
