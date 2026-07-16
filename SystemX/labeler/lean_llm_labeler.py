import concurrent.futures
import logging

import networkx as nx
from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.labeling.models import LeanLabelledNode
from SystemX.llm.config import Config
from SystemX.llm.llm_policy import ExecutionPolicy
from SystemX.llm.llm_provider import StructuredLLMProvider
from SystemX.prompts.LeanNodeClassificationTemplate import LeanNodeClassificationTemplate
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation

logger = logging.getLogger(__name__)

_CALL_NODE_TYPES = frozenset({9, 6})

_PREDICTION_CACHE_ENABLED = False
_PREDICTION_CACHE: dict[tuple[str, str], LeanLabelledNode] = {}

def enable_prediction_cache(enabled: bool = True) -> None:
    global _PREDICTION_CACHE_ENABLED
    _PREDICTION_CACHE_ENABLED = enabled

def clear_prediction_cache() -> None:
    _PREDICTION_CACHE.clear()

class LeanLLMLabeler(EdgeLabeler):
    """Zero-shot LLM hub classifier with no support and no reasoning output."""

    def __init__(
        self,
        config: Config,
        provider: StructuredLLMProvider,
        policy: ExecutionPolicy,
        template: LeanNodeClassificationTemplate | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.policy = policy
        self.template = template or LeanNodeClassificationTemplate()

    def apply_labels(self, graph: CDFIntermediateRepresentation) -> None:
        nx_G = graph.get_graph()
        system = self.template.render_system_message()

        targets = [(n, d) for n, d in nx_G.nodes(data=True) if d.get("node_type") in _CALL_NODE_TYPES]
        if not targets:
            return

        node_updates: dict = {}

        model_name = getattr(self.config, "model_name", "")

        def classify(node_id: str, data: dict) -> tuple[str, LeanLabelledNode | None]:
            if self.policy.stop_event.is_set():
                return node_id, None
            node_code = data.get("code") or data.get("label") or str(node_id)
            cache_key = (model_name, node_code)
            if _PREDICTION_CACHE_ENABLED and cache_key in _PREDICTION_CACHE:
                return node_id, _PREDICTION_CACHE[cache_key]
            self.policy.wait_for_slot()
            try:
                response = self.provider.generate(
                    prompt=self.template.render_user_message(node_code=node_code),
                    response_schema=LeanLabelledNode,
                    system_instruction=system,
                )
            except Exception as exc:
                logger.warning("LLM classification failed for node %s: %s", node_id, exc)
                return node_id, None
            self.policy.record_usage(response.token_usage)
            if _PREDICTION_CACHE_ENABLED:
                _PREDICTION_CACHE[cache_key] = response.data
            return node_id, response.data

        max_workers = max(1, getattr(self.config, "max_workers", 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(classify, n, d) for n, d in targets]
            for future in concurrent.futures.as_completed(futures):
                node_id, result = future.result()
                if result is None:
                    continue
                label_int = result.predicted_label
                node_updates[node_id] = {
                    "predicted_label": label_int,
                    "domain_label": REVERSE_DOMAIN_EDGE_TYPES.get(label_int, "NOT_RELEVANT"),
                    "predicted_confidence": 1.0,
                    "predicted_margin": 1.0,
                }

        nx.set_node_attributes(nx_G, node_updates)
        logger.info(
            "LeanLLMLabeler labelled %d/%d hub nodes (tokens used: %d).",
            len(node_updates),
            len(targets),
            self.policy.total_used,
        )
