import concurrent.futures
import logging
import random
import re
import threading
import time
from collections.abc import Callable

import networkx as nx

from SystemX.labeling.graph_service import SubgraphExtractor
from SystemX.labeling.models import MultiEdge, MultiLabelledNode
from SystemX.labeling.vamsa_kb_index import VamsaKBIndex
from SystemX.labeling.vamsa_loader import DomainEdgeId, VamsaEntry
from SystemX.llm.config import Config
from SystemX.llm.llm_policy import ExecutionPolicy
from SystemX.llm.llm_provider import StructuredLLMProvider
from SystemX.prompts.NodeClassificationTemplate import NodeClassificationTemplate
from SystemX.sca.constants import REVERSE_DOMAIN_EDGE_TYPES

logger = logging.getLogger(__name__)

class SystemXGraphLabeler:
    """High-concurrency, policy-enforced labeling engine for MultiDiGraphs."""

    def __init__(
        self,
        config: Config,
        graph: nx.MultiDiGraph,
        raw_code: str,
        provider: StructuredLLMProvider,
        policy: ExecutionPolicy,
        template: NodeClassificationTemplate,
        vamsa_mapping: dict[VamsaEntry, DomainEdgeId],
    ) -> None:
        self.config = config
        self.G = graph
        self.code_lines = raw_code.splitlines()
        self.provider = provider
        self.policy = policy
        self.template = template
        self._kb = VamsaKBIndex(vamsa_mapping)

        logger.info("Initialized SystemXGraphLabeler. Strategy: %s", "Batch" if config.batch_size > 1 else "Atomic")

    def _get_code_context(self, subgraph_edges: list[MultiEdge]) -> str:
        lines: set[int] = {line for edge in subgraph_edges if edge.lineno for line in edge.lineno}

        if not lines or -1 in lines:
            return "\n".join(self.code_lines)

        expanded_lines = sorted({line + offset for line in lines for offset in (-1, 0, 1) if 0 <= (line + offset) < len(self.code_lines)})
        return "\n".join(self.code_lines[i] for i in expanded_lines)

    def _parse_retry_after(self, error_msg: str) -> float:
        match = re.search(r"retry in ([\d.]+)s", error_msg)
        return float(match.group(1)) if match else self.config.retry_delay

    def _execute_with_policy(self, task: Callable[..., object], *args: object) -> object | None:
        thread_id = threading.get_ident()

        for attempt in range(self.config.retry_attempts):
            if self.policy.stop_event.is_set():
                logger.warning("[Thread %s] Execution aborted: Policy stop_event is set.", thread_id)
                return None

            self.policy.wait_for_slot()

            try:
                return task(*args)
            except Exception as e:
                err_str = str(e)
                retryable_terms = ("429", "RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED", "503", "UNAVAILABLE")
                is_retryable = any(x in err_str for x in retryable_terms)

                if is_retryable and attempt < self.config.retry_attempts - 1:
                    server_hint = self._parse_retry_after(err_str)
                    wait_time = max(server_hint, self.config.retry_delay * (2**attempt))
                    wait_time += random.uniform(0.5, 1.5)
                    logger.warning("[Thread %s] Retryable error. Backing off for %.2fs...", thread_id, wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("[Thread %s] Fatal or terminal error: %s", thread_id, e)
                    break
        return None

    def _label_node_worker(self, node_id: str, data: dict) -> MultiLabelledNode | None:
        def action() -> MultiLabelledNode:
            subgraph = SubgraphExtractor.extract(self.G, node_id, max_depth=self.config.max_depth)
            node_code = data.get("code", "")

            response = self.provider.generate(
                prompt=self.template.render_user_message(
                    node_id=node_id,
                    node_code=node_code,
                    subgraph_context=str(subgraph),
                    raw_code=self._get_code_context(subgraph.edges),
                ),
                system_instruction=self.template.render_system_message(),
                response_schema=MultiLabelledNode,
            )
            self.policy.record_usage(response.token_usage)
            return response.data

        return self._execute_with_policy(action)

    def label_graph(self, batch_size: int = 1) -> tuple[nx.DiGraph, int]:
        target_nodes = [(n, d) for n, d in self.G.nodes(data=True) if d.get("node_type") in (9, 6) and "domain_label" not in d]

        if not target_nodes:
            return self.G, self.policy.total_used

        unmatched_nodes = []
        for node_id, data in target_nodes:
            static_match = self._kb.match(node_id, data)
            if static_match:
                self._apply_node_label(node_id, static_match)
            else:
                unmatched_nodes.append((node_id, data))

        if not unmatched_nodes:
            logger.info("All nodes resolved via static KB. Bypassing LLM entirely.")
            return self.G, self.policy.total_used

        logger.info(
            "KB resolved %d nodes. Falling back to LLM for %d nodes.",
            len(target_nodes) - len(unmatched_nodes),
            len(unmatched_nodes),
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self._label_node_worker, n, d): n for n, d in unmatched_nodes}
            for future in concurrent.futures.as_completed(futures):
                node_id = futures[future]
                self._apply_node_label(node_id, future.result())

        return self.G, self.policy.total_used

    def _apply_node_label(self, node_id: str, result: MultiLabelledNode | None) -> None:
        if result:
            predicted = result.predicted_label
            self.G.nodes[node_id].update(
                {
                    "domain_label": REVERSE_DOMAIN_EDGE_TYPES[predicted],
                    "predicted_label": predicted,
                }
            )
