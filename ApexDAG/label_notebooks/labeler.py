import concurrent.futures
import logging
import time

import networkx as nx

from ApexDAG.label_notebooks.graph_service import SubgraphExtractor
from ApexDAG.label_notebooks.llm_policy import ExecutionPolicy
from ApexDAG.label_notebooks.message_template import (
    generate_system_prompt,
    generate_user_message,
)
from ApexDAG.label_notebooks.schema import BatchLabelResponse, MultiLabelledEdge
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.llm.llm_provider import StructuredLLMProvider

logger = logging.getLogger(__name__)


class ApexGraphLabeler:
    """
    Concurrent, structured-output LLM labeling engine for MultiDiGraphs.
    Delegates LLM execution to a Provider and budget tracking to a Policy.
    """

    def __init__(self, config: Config, graph: nx.MultiDiGraph, raw_code: str, provider: StructuredLLMProvider, policy: ExecutionPolicy) -> None:
        self.config = config
        self.G = graph
        self.code_lines = raw_code.splitlines()
        self.provider = provider
        self.policy = policy

        logger.info(f"Initialized ApexGraphLabeler. Budget: {self.policy.max_tokens} tokens.")

    def _get_code_context(self, subgraph_edges: list) -> str:
        """Extracts relevant code snippets for the LLM to analyze."""
        lines = {line for edge in subgraph_edges if edge.lineno for line in edge.lineno}

        if not lines or -1 in lines:
            return "\n".join(self.code_lines)

        expanded_lines = sorted({line + offset for line in lines for offset in (-1, 0, 1) if 0 <= (line + offset) < len(self.code_lines)})
        return "\n".join(self.code_lines[i] for i in expanded_lines)

    def _label_edge_worker(self, src: str, tgt: str, key: str, edge_data: dict, max_depth: int) -> MultiLabelledEdge | None:
        if self.policy.stop_event.is_set():
            return None

        self.policy.wait_for_slot()
        for attempt in range(self.config.retry_attempts):
            try:
                subgraph = SubgraphExtractor.extract(self.G, src, tgt, key, max_depth=max_depth)
                code_context = self._get_code_context(subgraph.edges)

                response = self.provider.generate(
                    prompt=generate_user_message(
                        source_id=src,
                        target_id=tgt,
                        edge_code=str(edge_data.get("code", "")),
                        subgraph_context=str(subgraph),
                        raw_code=code_context,
                    ),
                    system_instruction=generate_system_prompt(),
                    response_schema=MultiLabelledEdge,
                )

                self.policy.record_usage(response.token_usage)
                return response.data

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {src}->{tgt}: {e}")
                # Exponential backoff on rate limits
                delay = self.config.retry_delay * (2 if "429" in str(e) else 1)
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(delay)
                else:
                    logger.error(f"Fatal error labeling {src}->{tgt} after {attempt + 1} attempts.")
                    return None
        return None

    def _label_batch_worker(self, batch_tasks: list[tuple[str, str, str, dict]]) -> list[tuple[str, str, str, MultiLabelledEdge | None]]:
        """Processes a chunk of edges in a single LLM call."""
        if self.policy.stop_event.is_set():
            return []

        # Prepare combined context
        batch_prompts = []
        for src, tgt, key, data in batch_tasks:
            subgraph = SubgraphExtractor.extract(self.G, src, tgt, key, max_depth=self.config.max_depth)
            ctx = self._get_code_context(subgraph.edges)
            # Minimal prompt per edge to save tokens in the combined block
            batch_prompts.append(f"Edge ID: {key}\nCode: {data.get('code')}\nContext: {ctx}")

        aggregated_prompt = "\n---\n".join(batch_prompts)
        system_instr = generate_system_prompt() + "\nReturn a JSON array of labels corresponding to the Edge IDs provided."

        try:
            response = self.provider.generate(prompt=aggregated_prompt, system_instruction=system_instr, response_schema=BatchLabelResponse)
            self.policy.record_usage(response.token_usage)

            # Map results back to IDs
            label_map = {item.edge_id: item.label for item in response.data.labels}
            return [(u, v, k, label_map.get(k)) for u, v, k, d in batch_tasks]

        except Exception as e:
            logger.error(f"Batch inference failure: {e}")
            return [(u, v, k, None) for u, v, k, d in batch_tasks]

    def label_graph(self, batch_size: int = 1) -> tuple[nx.MultiDiGraph, int]:
        """
        Orchestrates labeling.
        If batch_size > 1, uses the aggregated method. Otherwise, uses atomic.
        """
        edges = list(self.G.edges(data=True, keys=True))

        if batch_size <= 1:
            return self._run_atomic(edges)

        # Batching Logic
        chunks = [edges[i : i + batch_size] for i in range(0, len(edges), batch_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self._label_batch_worker, chunk) for chunk in chunks]

            for future in concurrent.futures.as_completed(futures):
                results = future.result()
                for u, v, key, labeled_edge in results:
                    self._apply_label(u, v, key, labeled_edge)

        return self.G, self.policy.total_used

    def _apply_label(self, u: str, v: str, key: str, labeled_edge: MultiLabelledEdge | None) -> None:
        if labeled_edge:
            self.G.edges[u, v, key].update({"domain_label": labeled_edge.domain_label, "reasoning": labeled_edge.reasoning})
        else:
            self.G.edges[u, v, key].update({"domain_label": "MISSING", "reasoning": "Inference failure."})
