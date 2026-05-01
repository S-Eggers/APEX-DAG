import logging

from ApexDAG.label_notebooks.labeler import ApexGraphLabeler
from ApexDAG.label_notebooks.llm_policy import ExecutionPolicy
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.labeler.edge_labeler import EdgeLabeler
from ApexDAG.llm.llm_provider import StructuredLLMProvider
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph

logger = logging.getLogger(__name__)


class LLMLabeler(EdgeLabeler):
    """
    A stateless edge labeler that delegates execution to the ApexGraphLabeler.
    Requires external injection of Provider and Policy to maintain global constraints.
    """

    def __init__(self) -> None:
        self.config: Config | None = None
        self.provider: StructuredLLMProvider | None = None
        self.policy: ExecutionPolicy | None = None

    def configure(self, config: Config, provider: StructuredLLMProvider, policy: ExecutionPolicy) -> None:
        """Inject dependencies required for execution."""
        self.config = config
        self.provider = provider
        self.policy = policy

    def apply_labels(self, graph: PythonDataFlowGraph) -> None:
        if not self.config or not self.provider or not self.policy:
            raise RuntimeError("LLMLabeler must be configured with a Config, Provider, and Policy before use.")

        logger.info(f"Starting LLM labeling for graph with {len(graph.get_graph().edges)} edges.")

        labeler = ApexGraphLabeler(config=self.config, graph=graph.get_graph(), raw_code=graph.get_code(), provider=self.provider, policy=self.policy)

        batch_size = getattr(self.config, "batch_size", 25)
        labeled_graph, tokens_used = labeler.label_graph(batch_size=batch_size)

        attrs_to_set = {}
        for u, v, key, data in labeled_graph.edges(data=True, keys=True):
            domain_label = data.get("domain_label", "NOT_RELEVANT").upper()
            attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES.get(domain_label, DOMAIN_EDGE_TYPES["NOT_RELEVANT"])

        graph.set_domain_label(attrs_to_set, name="predicted_label")
        logger.info(f"Successfully applied labels. Tokens consumed in this session: {tokens_used}")
