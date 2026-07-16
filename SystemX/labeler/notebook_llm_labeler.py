import logging

import networkx as nx
from SystemX.labeler.context_llm_labeler import _CALL_NODE_TYPES, _code, _topological_context
from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.labeling.models import NotebookClassification
from SystemX.llm.config import Config
from SystemX.llm.llm_policy import ExecutionPolicy
from SystemX.llm.llm_provider import StructuredLLMProvider
from SystemX.prompts.StrongNotebookClassificationTemplate import StrongNotebookClassificationTemplate
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation

logger = logging.getLogger(__name__)

class NotebookLLMLabeler(EdgeLabeler):
    """Strong LLM baseline: whole-notebook, few-shot, chain-of-thought."""

    def __init__(
        self,
        config: Config,
        provider: StructuredLLMProvider,
        policy: ExecutionPolicy,
        template: StrongNotebookClassificationTemplate | None = None,
        max_nodes_per_call: int = 60,
    ) -> None:
        self.config = config
        self.provider = provider
        self.policy = policy
        self.template = template or StrongNotebookClassificationTemplate()
        self.max_nodes_per_call = max_nodes_per_call
        self.last_reasonings: dict[object, str] = {}

    def _serialize(self, nx_G: nx.MultiDiGraph, targets: list, id_of: dict) -> str:
        blocks = []
        for node_id in targets:
            data = nx_G.nodes[node_id]
            ctx = _topological_context(nx_G, node_id).replace("\n", "; ")
            blocks.append(f"[{id_of[node_id]}] {_code(data, node_id)}\n  context: {ctx}")
        return "\n".join(blocks)

    def apply_labels(self, graph: CDFIntermediateRepresentation) -> None:
        nx_G = graph.get_graph()
        targets = [n for n, d in nx_G.nodes(data=True) if d.get("node_type") in _CALL_NODE_TYPES]
        if not targets:
            return

        system = self.template.render_system_message()
        node_updates: dict = {}
        self.last_reasonings = {}

        for start in range(0, len(targets), self.max_nodes_per_call):
            if self.policy.stop_event.is_set():
                break
            chunk = targets[start : start + self.max_nodes_per_call]
            id_of = {n: f"op{start + i}" for i, n in enumerate(chunk)}
            node_of = {v: k for k, v in id_of.items()}

            self.policy.wait_for_slot()
            try:
                response = self.provider.generate(
                    prompt=self.template.render_user_message(self._serialize(nx_G, chunk, id_of)),
                    response_schema=NotebookClassification,
                    system_instruction=system,
                )
            except Exception as exc:
                logger.warning("Strong LLM classification failed for a chunk: %s", exc)
                continue
            self.policy.record_usage(response.token_usage)

            for item in response.data.nodes:
                real = node_of.get(str(item.node_id).strip())
                if real is None:
                    continue
                node_updates[real] = {
                    "predicted_label": item.predicted_label,
                    "domain_label": REVERSE_DOMAIN_EDGE_TYPES.get(item.predicted_label, "NOT_RELEVANT"),
                    "predicted_confidence": 1.0,
                    "predicted_margin": 1.0,
                }
                self.last_reasonings[real] = item.reasoning

        nx.set_node_attributes(nx_G, node_updates)
        logger.info(
            "NotebookLLMLabeler labelled %d/%d hub nodes (tokens used: %d).",
            len(node_updates), len(targets), self.policy.total_used,
        )
