import concurrent.futures
import logging
import random
import re
import threading
import time
from collections.abc import Callable

import networkx as nx

from ApexDAG.label_notebooks.graph_service import SubgraphExtractor
from ApexDAG.label_notebooks.llm_policy import ExecutionPolicy
from ApexDAG.label_notebooks.models import BatchLabelResponse, MultiEdge, MultiLabelledEdge
from ApexDAG.llm.llm_provider import StructuredLLMProvider
from ApexDAG.llm.models import Config
from ApexDAG.prompts.EdgeClassificationTemplate import EdgeClassificationTemplate

logger = logging.getLogger(__name__)


class ApexGraphLabeler:
    """
    A high-concurrency, policy-enforced labeling engine for MultiDiGraphs.
    Uses a centralized ExecutionPolicy to manage RPM and token quotas.
    """

    def __init__(self, config: Config, graph: nx.MultiDiGraph, raw_code: str, provider: StructuredLLMProvider, policy: ExecutionPolicy, template: EdgeClassificationTemplate) -> None:
        self.config = config
        self.G = graph
        self.code_lines = raw_code.splitlines()
        self.provider = provider
        self.policy = policy
        self.template = template

        logger.info(f"Initialized ApexGraphLabeler. Strategy: {'Batch' if config.batch_size > 1 else 'Atomic'}")

    def _get_code_context(self, subgraph_edges: list[MultiEdge]) -> str:
        """Extracts localized code context with a 1-line buffer."""
        lines: set[int] = {line for edge in subgraph_edges if edge.lineno for line in edge.lineno}

        if not lines or -1 in lines:
            return "\n".join(self.code_lines)

        expanded_lines = sorted({line + offset for line in lines for offset in (-1, 0, 1) if 0 <= (line + offset) < len(self.code_lines)})
        return "\n".join(self.code_lines[i] for i in expanded_lines)

    def _parse_retry_after(self, error_msg: str) -> float:
        """Extracts the recommended wait time from Google's 429 response."""
        match = re.search(r"retry in ([\d.]+)s", error_msg)
        return float(match.group(1)) if match else self.config.retry_delay

    def _execute_with_policy(self, task: Callable[..., object], *args: object) -> object | None:
        thread_id = threading.get_ident()

        for attempt in range(self.config.retry_attempts):
            if self.policy.stop_event.is_set():
                logger.warning(f"[Thread {thread_id}] Execution aborted: Policy stop_event is set.")
                return None

            self.policy.wait_for_slot()

            try:
                logger.debug(f"[Thread {thread_id}] Attempting task (Attempt {attempt + 1})")
                return task(*args)
            except Exception as e:
                err_str = str(e)
                retryable_terms = ["429", "RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED", "503", "UNAVAILABLE"]
                is_retryable = any(x in err_str for x in retryable_terms)

                if is_retryable and attempt < self.config.retry_attempts - 1:
                    server_hint = self._parse_retry_after(err_str)
                    wait_time = max(server_hint, self.config.retry_delay * (2**attempt))
                    wait_time += random.uniform(0.5, 1.5)

                    logger.warning(f"[Thread {thread_id}] Retryable error: {err_str[:100]}. Backing off for {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[Thread {thread_id}] Fatal or terminal error: {e}")
                    break
        return None

    def _label_edge_worker(self, src: str, tgt: str, key: str, data: dict[str, object]) -> MultiLabelledEdge | None:
        """Atomic worker for processing a single edge."""

        def action() -> MultiLabelledEdge:
            subgraph = SubgraphExtractor.extract(self.G, src, tgt, key, max_depth=self.config.max_depth)
            code_context = self._get_code_context(subgraph.edges)

            response = self.provider.generate(
                prompt=self.template.render_user_message(
                    source_id=src,
                    target_id=tgt,
                    edge_code=str(data.get("code", "")),
                    subgraph_context=str(subgraph),
                    raw_code=code_context,
                ),
                system_instruction=self.template.render_system_message(),
                response_schema=MultiLabelledEdge,
            )
            self.policy.record_usage(response.token_usage)
            return response.data

        return self._execute_with_policy(action)

    def _label_batch_worker(self, batch_tasks: list[tuple[str, str, str, dict[str, object]]]) -> list[tuple[str, str, str, MultiLabelledEdge | None]]:
        """Worker for processing aggregated edges to optimize token/RPM ratio."""

        def action() -> list[tuple[str, str, str, MultiLabelledEdge | None]]:
            batch_prompts = []
            for src, tgt, key, data in batch_tasks:
                subgraph = SubgraphExtractor.extract(self.G, src, tgt, key, max_depth=self.config.max_depth)
                ctx = self._get_code_context(subgraph.edges)
                batch_prompts.append(f"Edge ID: {key}\nCode: {data.get('code')}\nContext: {ctx}")

            aggregated_prompt = "\n---\n".join(batch_prompts)
            system_instr = f"{self.template.render_system_message()}\nReturn a JSON array of labels corresponding to the Edge IDs provided."

            response = self.provider.generate(prompt=aggregated_prompt, system_instruction=system_instr, response_schema=BatchLabelResponse)
            self.policy.record_usage(response.token_usage)

            label_map = {item.edge_id: item.label for item in response.data.labels}
            return [(u, v, k, label_map.get(k)) for u, v, k, d in batch_tasks]

        results = self._execute_with_policy(action)
        return results if results else [(u, v, k, None) for u, v, k, d in batch_tasks]

    def label_graph(self, batch_size: int = 1) -> tuple[nx.MultiDiGraph, int]:
        """
        Orchestrates the labeling process. Uses ThreadPoolExecutor to handle
        the I/O wait times while respecting the ExecutionPolicy.
        """
        edges = list(self.G.edges(data=True, keys=True))

        unlabelled = [(u, v, k, d) for u, v, k, d in edges if "domain_label" not in d]

        if not unlabelled:
            return self.G, self.policy.total_used

        if batch_size <= 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self._label_edge_worker, u, v, k, d): (u, v, k) for u, v, k, d in unlabelled}
                for future in concurrent.futures.as_completed(futures):
                    u, v, k = futures[future]
                    self._apply_label(u, v, k, future.result())
        else:
            chunks = [unlabelled[i : i + batch_size] for i in range(0, len(unlabelled), batch_size)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(self._label_batch_worker, chunk) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    for u, v, k, labeled_edge in future.result():
                        self._apply_label(u, v, k, labeled_edge)

        return self.G, self.policy.total_used

    def _apply_label(self, u: str, v: str, key: str, labeled_edge: MultiLabelledEdge | None) -> None:
        if labeled_edge:
            logger.debug(f"Applying label to edge {key}: {labeled_edge.domain_label}")
            self.G.edges[u, v, key].update({"domain_label": labeled_edge.domain_label, "reasoning": labeled_edge.reasoning})
        else:
            logger.warning(f"Edge {key} marked as NOT_RELEVANT due to processing failure.")
            self.G.edges[u, v, key].update({"domain_label": "NOT_RELEVANT", "reasoning": "Inference failure."})
