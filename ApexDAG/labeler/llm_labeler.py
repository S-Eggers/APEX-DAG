import logging

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


from ApexDAG.label_notebooks.__main__ import get_provider
from ApexDAG.label_notebooks.labeler import ApexGraphLabeler
from ApexDAG.label_notebooks.token_policy import TokenBudgetPolicy
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.labeler.edge_labeler import EdgeLabeler
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class LLMLabeler(EdgeLabeler):
    def apply_labels(self, graph: PythonDataFlowGraph) -> None:
        config = Config(
            model_name="gemini-3.1-flash-lite-preview",
            max_tokens=float("inf"),
            max_depth=4,
            llm_provider="google",
            retry_attempts=2,
            retry_delay=1,
            success_delay=0,
            sleep_interval=0,
            max_workers=1,
        )
        provider = get_provider(config)
        global_budget = TokenBudgetPolicy(max_tokens=config.max_tokens)

        labeler = ApexGraphLabeler(config=config, graph=graph.get_graph(), raw_code=graph.get_code(), provider=provider, token_budget=global_budget)

        labeled_graph, tokens_used = labeler.label_graph(batch_size=25)
        logger.info(f"LLMLabeler complete. Total tokens consumed: {tokens_used}")

        attrs_to_set = {}
        for u, v, key, data in labeled_graph.edges(data=True, keys=True):
            domain_label = data.get("domain_label", "NOT_RELEVANT").upper()

            if domain_label in DOMAIN_EDGE_TYPES:
                attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES[domain_label]
            else:
                attrs_to_set[(u, v, key)] = DOMAIN_EDGE_TYPES["NOT_RELEVANT"]

        graph.set_domain_label(attrs_to_set, name="predicted_label")
