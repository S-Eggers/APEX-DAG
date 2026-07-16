import concurrent.futures
import logging

import networkx as nx
from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.labeling.models import LeanLabelledNode
from SystemX.llm.config import Config
from SystemX.llm.llm_policy import ExecutionPolicy
from SystemX.llm.llm_provider import StructuredLLMProvider
from SystemX.prompts.NodeClassificationTemplate import NodeClassificationTemplate
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation

logger = logging.getLogger(__name__)

_CALL_NODE_TYPES = frozenset({9, 6})

_VARIABLE = 0
_MODULE = 1
_SINK = 4

def _short(data: dict, node_id: str) -> str:
    """Short human name for a variable/module node (its label, not its full defn code)."""
    return str(data.get("label") or data.get("code") or node_id).strip()

def _code(data: dict, node_id: str) -> str:
    """Call-expression code for an operation node (what the consumer actually does)."""
    return str(data.get("code") or data.get("label") or node_id).strip()

def _topological_context(nx_G: nx.MultiDiGraph, node_id: str) -> str:
    """Serialize the node's dataflow neighborhood for the LLM - correctly."""
    preds = list(nx_G.predecessors(node_id))
    pred_ids = set(preds)

    data_inputs = [_short(nx_G.nodes[u], u) for u in preds if nx_G.nodes[u].get("node_type") == _VARIABLE]
    libs = [_short(nx_G.nodes[u], u) for u in preds if nx_G.nodes[u].get("node_type") == _MODULE]

    new_vars: list[str] = []
    read_only: list[str] = []
    side_effect = False
    consumers: list[str] = []
    for v in nx_G.successors(node_id):
        vt = nx_G.nodes[v].get("node_type")
        if vt == _SINK:
            side_effect = True
        elif vt == _VARIABLE:
            if v in pred_ids:
                read_only.append(_short(nx_G.nodes[v], v))
            else:
                new_vars.append(_short(nx_G.nodes[v], v))
                for w in nx_G.successors(v):
                    if nx_G.nodes[w].get("node_type") in _CALL_NODE_TYPES:
                        consumers.append(_code(nx_G.nodes[w], w))

    lines = [f"Data inputs: {', '.join(dict.fromkeys(data_inputs)) or '(none)'}"]
    if libs:
        lines.append(f"Library/module: {', '.join(dict.fromkeys(libs))}")
    if new_vars:
        lines.append(f"Assigns result to: {', '.join(dict.fromkeys(new_vars))}")
        if consumers:
            uniq = list(dict.fromkeys(consumers))[:8]
            lines.append(f"Result is then used by later operation(s): {'; '.join(uniq)}")
        else:
            lines.append("Result is NOT used by any later operation (dead-end: only displayed/printed or discarded).")
    elif read_only:
        lines.append(
            f"Reads/inspects existing variable(s) '{', '.join(dict.fromkeys(read_only))}' in place; "
            "produces no distinct value fed downstream."
        )
    elif side_effect:
        lines.append("Side-effect only - produces no value into the dataflow (e.g. plotting, printing, config).")
    else:
        lines.append("Produces no tracked output.")
    return "\n".join(lines)

class ContextLLMLabeler(EdgeLabeler):
    """Context-aware LLM hub classifier - the "rich" counterpart of the zero-shot ~SystemX.labeler.lean_llm_labeler.LeanLLMLabeler."""

    def __init__(
        self,
        config: Config,
        provider: StructuredLLMProvider,
        policy: ExecutionPolicy,
        template: NodeClassificationTemplate | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.policy = policy
        self.template = template or NodeClassificationTemplate()

    def apply_labels(self, graph: CDFIntermediateRepresentation) -> None:
        nx_G = graph.get_graph()
        system = self.template.render_system_message()

        targets = [(n, d) for n, d in nx_G.nodes(data=True) if d.get("node_type") in _CALL_NODE_TYPES]
        if not targets:
            return

        node_updates: dict = {}

        def classify(node_id: str, data: dict) -> tuple[str, LeanLabelledNode | None]:
            if self.policy.stop_event.is_set():
                return node_id, None
            node_code = _code(data, node_id)
            context = _topological_context(nx_G, node_id)
            self.policy.wait_for_slot()
            try:
                response = self.provider.generate(
                    prompt=self.template.render_user_message(
                        node_id=node_id, node_code=node_code, subgraph_context=context, raw_code=""
                    ),
                    response_schema=LeanLabelledNode,
                    system_instruction=system,
                )
            except Exception as exc:
                logger.warning("Context LLM classification failed for node %s: %s", node_id, exc)
                return node_id, None
            self.policy.record_usage(response.token_usage)
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
            "ContextLLMLabeler labelled %d/%d hub nodes (tokens used: %d).",
            len(node_updates),
            len(targets),
            self.policy.total_used,
        )
