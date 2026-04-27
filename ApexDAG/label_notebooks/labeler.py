import concurrent.futures
import json
import logging
import os
import threading
import time

import networkx as nx
from dotenv import load_dotenv

from ApexDAG.label_notebooks.graph_service import SubgraphExtractor
from ApexDAG.label_notebooks.message_template import (
    generate_system_prompt,
    generate_user_message,
)
from ApexDAG.label_notebooks.schema import MultiGraphContext, MultiLabelledEdge
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class ApexGraphLabeler:
    """
    Concurrent, structured-output LLM labeling engine for MultiDiGraphs.
    Includes a Thread-Safe Token Budget to prevent API financial ruin.
    """

    def __init__(self, config: Config, graph: nx.MultiDiGraph, raw_code: str) -> None:
        load_dotenv()
        self.config = config
        self.G = graph
        self.code_lines = raw_code.splitlines()
        self.llm_provider = getattr(config, "llm_provider", "google")

        # Concurrency & Budget Safety Net
        self.max_tokens = getattr(self.config, "max_tokens", float("inf"))
        self.total_tokens_used = 0
        self.token_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.base_context = MultiGraphContext.from_graph(self.G)
        self.client = self._initialize_client()

        logger.info(
            f"""Initialized ApexGraphLabeler ({self.llm_provider}).
            Budget: {self.max_tokens} tokens."""
        )

    def _initialize_client(self) -> None:
        if self.llm_provider == "google":
            import google.generativeai as genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.config.model_name)

        elif self.llm_provider == "groq":
            import instructor
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            return instructor.from_groq(
                Groq(api_key=api_key), mode=instructor.Mode.TOOLS
            )

        raise ValueError(f"Unsupported LLM provider: '{self.llm_provider}'.")

    def _create_structured_completion(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[MultiLabelledEdge, int]:
        """
        Executes the API call, returning the strictly typed label and
        the tokens consumed.
        """
        if self.llm_provider == "google":
            import google.generativeai as genai

            response = self.client.generate_content(
                contents=[
                    {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
                ],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=MultiLabelledEdge,
                    temperature=0.1,
                ),
            )
            tokens = getattr(response.usage_metadata, "total_token_count", 0)
            return MultiLabelledEdge(**json.loads(response.text)), tokens

        elif self.llm_provider == "groq":
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                response_model=MultiLabelledEdge,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            tokens = (len(system_prompt) + len(user_prompt)) // 4 + 50
            return response, tokens

    def _get_code_context(self, subgraph_edges: list) -> str:
        lines = set()
        for edge in subgraph_edges:
            if edge.lineno:
                lines.update(edge.lineno)

        if -1 in lines or not lines:
            return "\n".join(self.code_lines)

        expanded_lines = sorted(
            {
                line + offset
                for line in lines
                for offset in (-1, 0, 1)
                if 0 <= (line + offset) < len(self.code_lines)
            }
        )
        return "\n".join(self.code_lines[i] for i in expanded_lines)

    def _label_edge_worker(
        self, src: str, tgt: str, key: str, edge_data: dict, max_depth: int
    ) -> MultiLabelledEdge | None:
        if self.stop_event.is_set():
            return None

        for attempt in range(self.config.retry_attempts):
            try:
                subgraph = SubgraphExtractor.extract(
                    self.G, src, tgt, key, max_depth=max_depth
                )
                code_context = self._get_code_context(subgraph.edges)

                sys_prompt = generate_system_prompt()
                usr_prompt = generate_user_message(
                    source_id=src,
                    target_id=tgt,
                    edge_code=str(edge_data.get("code", "")),
                    subgraph_context=str(subgraph),
                    raw_code=code_context,
                )

                labeled_edge, tokens_used = self._create_structured_completion(
                    sys_prompt, usr_prompt
                )

                with self.token_lock:
                    self.total_tokens_used += tokens_used
                    if self.total_tokens_used >= self.max_tokens:
                        logger.warning(
                            f"""
BUDGET EXCEEDED! {self.total_tokens_used} tokens used. Halting pool."""
                        )
                        self.stop_event.set()

                return labeled_edge

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {src}->{tgt}: {e}")
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(self.config.retry_delay * 2)
                elif attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Fatal error on {src}->{tgt}: {e}")
                    return None

        return None

    def label_graph(self) -> nx.MultiDiGraph:
        edges_to_process = list(self.G.edges(data=True, keys=True))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            future_to_edge = {
                executor.submit(
                    self._label_edge_worker, u, v, key, data, self.config.max_depth
                ): (u, v, key)
                for u, v, key, data in edges_to_process
            }

            for future in concurrent.futures.as_completed(future_to_edge):
                u, v, key = future_to_edge[future]

                labeled_edge = future.result()
                if labeled_edge is None:
                    self.G.edges[u, v, key]["domain_label"] = "MISSING"
                    self.G.edges[u, v, key]["reasoning"] = (
                        "Budget limit reached or LLM failure."
                    )
                else:
                    self.G.edges[u, v, key]["domain_label"] = labeled_edge.domain_label
                    self.G.edges[u, v, key]["reasoning"] = labeled_edge.reasoning

        return self.G, self.total_tokens_used
