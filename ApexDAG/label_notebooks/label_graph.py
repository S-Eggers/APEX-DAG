import os
import re
import json
import time
import logging
import instructor
import networkx as nx
from tqdm import tqdm
from typing import Literal
from dotenv import load_dotenv
from pydantic import parse_obj_as, TypeAdapter, ValidationError

from ApexDAG.sca.graph_utils import load_graph
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.label_notebooks.message_template import generate_message
from ApexDAG.label_notebooks.pydantic_models import (
    LabelledEdge,
    GraphContextWithSubgraphSearch,
    SubgraphContext,
)


class TokenLimitExceededError(Exception):
    """Custom exception raised when the configured token limit is reached."""

    pass


class GraphLabeler:
    """
    Labels graph edges using an LLM, with support for multiple providers and token counting.
    """

    def __init__(self, config: Config, graph_path: str, code_path: str, logger=None):
        load_dotenv()
        self.config = config
        self.graph_path = graph_path
        self.G = load_graph(self.graph_path)

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        self.llm_provider = getattr(config, "llm_provider", "groq")
        self.client = self._initialize_client()

        self.G_with_context = GraphContextWithSubgraphSearch.from_graph(self.G)
        self.code_lines = self.read_code(code_path)

        self.total_tokens_used = 0
        self.logger.info(
            f"Initialized GraphLabeler with provider '{self.llm_provider}' and model '{self.config.model_name}'."
        )
        if hasattr(self.config, "max_tokens") and self.config.max_tokens > 0:
            self.logger.info(f"Token limit set to {self.config.max_tokens} tokens.")
        else:
            self.logger.warning(
                "No max_tokens limit set in config. Running without a token budget."
            )
            self.config.max_tokens = float("inf")

    def _initialize_client(self):
        """
        Initializes and patches the LLM client, importing libraries only when needed.
        """
        self.logger.info(f"Initializing client for provider: {self.llm_provider}")
        if self.llm_provider == "groq":
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable not set for Groq provider."
                )
            client = Groq(api_key=api_key)
            return instructor.from_groq(client, mode=instructor.Mode.TOOLS)

        elif self.llm_provider == "google":
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError(
                    "google-generativeai is not installed. Please run: pip install google-generativeai"
                ) from e

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set for Google provider."
                )
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(self.config.model_name)
            return client

        else:
            raise ValueError(
                f"Unsupported LLM provider: '{self.llm_provider}'. Supported providers are 'groq' and 'google'."
            )

    def _create_chat_completion(self, messages, response_model):
        """
        Makes a generic API call to the configured LLM and returns the response and token usage.
        """
        if self.llm_provider == "groq":
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                response_model=response_model,
            )
            return response, response.usage

        elif self.llm_provider == "google":
            # Convert messages to Gemini format (list of content strings)
            gemini_messages = [
                msg["content"] for msg in messages if msg["role"] == "user"
            ]

            response = self.client.generate_content(
                gemini_messages,
                generation_config={"response_mime_type": "application/json"},
            )

            # Parse the JSON response and validate with Pydantic
            try:
                parsed_response = json.loads(response.text)
                validated_response = TypeAdapter(response_model).validate_python(
                    parsed_response
                )
            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.error(f"Error parsing or validating Gemini response: {e}")
                self.logger.error(f"Raw Gemini response: {response.text}")
                raise

            # Extract token usage
            usage_metadata = response.usage_metadata
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
            total_tokens = prompt_tokens + completion_tokens

            return validated_response, total_tokens

    def read_code(self, code_path: str):
        if not os.path.exists(code_path):
            raise FileNotFoundError(f"The file at '{code_path}' does not exist.")
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        return code.splitlines()

    def get_input_subgraph(
        self,
        node_id_source: str,
        node_id_target: str,
        max_depth: int = 1,
    ):
        subgraph_nodes, subgraph_edges = self.G_with_context.get_subgraph(
            node_id_source, node_id_target, max_depth=max_depth
        )
        model_input = SubgraphContext(
            edge_of_interest=(node_id_source, node_id_target),
            nodes=subgraph_nodes,
            edges=subgraph_edges,
        )
        return str(model_input)

    def get_input_code_context(
        self,
        node_id_source: str,
        node_id_target: str,
        max_depth: int = 1,
        allow_all_code: bool = True,
    ):
        _, subgraph_edges = self.G_with_context.get_subgraph(
            node_id_source, node_id_target, max_depth=max_depth
        )
        lines = sorted(
            {line for edge in subgraph_edges for line in (edge.lineno or [])}
        )
        if -1 in lines and allow_all_code:
            return "\n".join(self.code_lines)
        expanded_lines = sorted(
            {
                line_offset
                for line in lines
                for line_offset in (line - 1, line, line + 1)
                if 0 <= line_offset < len(self.code_lines)
            }
        )
        return "\n".join(self.code_lines[line] for line in expanded_lines)

    def extract_wait_time(self, error_message):
        match = re.search(r"Please try again in (\d+)m(\d+\.\d+)s", error_message)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        return None

    def label_edge(self, edge, edge_num_index, max_depth=2):
        if self.total_tokens_used >= self.config.max_tokens:
            raise TokenLimitExceededError(
                f"Token limit of {self.config.max_tokens} reached. "
                f"Current usage: {self.total_tokens_used}. Halting processing."
            )

        start_time = time.time()
        retry_attempts = self.config.retry_attempts
        retry_delay = self.config.retry_delay
        success_delay = self.config.success_delay
        allow_all_code = True
        resp = None

        for attempt in range(retry_attempts):
            try:
                graph_context = self.get_input_subgraph(
                    edge.source, edge.target, max_depth=max_depth
                )
                code_context = self.get_input_code_context(
                    edge.source,
                    edge.target,
                    max_depth=max_depth,
                    allow_all_code=allow_all_code,
                )
                messages = generate_message(
                    edge.source, edge.target, edge.code, graph_context, code_context
                )

                resp, usage = self._create_chat_completion(
                    messages=messages,
                    response_model=LabelledEdge,
                )

                if self.llm_provider == "groq":
                    tokens_this_call = usage.total_tokens
                elif self.llm_provider == "google":
                    tokens_this_call = usage  # usage is already total_tokens for google
                else:
                    # Fallback for unknown providers
                    prompt_tokens = len(messages[0]["content"]) / 4
                    completion_tokens = len(resp.model_dump_json()) / 4
                    tokens_this_call = prompt_tokens + completion_tokens

                self.total_tokens_used += tokens_this_call
                self.logger.info(
                    f"Tokens used for ({edge.source} -> {edge.target}): {tokens_this_call}. "
                    f"Total used: {self.total_tokens_used}/{self.config.max_tokens}"
                )

                if resp.domain_label != "MORE_CONTEXT_NEEDED":
                    time.sleep(success_delay)
                    break
                else:
                    self.logger.info(
                        "'MORE_CONTEXT_NEEDED' received. Retrying with increased depth."
                    )
                    time.sleep(retry_delay)
                    max_depth += 1
            except instructor.exceptions.InstructorRetryException as e:
                self.logger.error(
                    f"Rate limit error on edge {edge.source} -> {edge.target}: {e}"
                )
                if getattr(e, "status_code", None) == 413:
                    allow_all_code = False
                wait_time = self.extract_wait_time(str(e))
                if wait_time:
                    retry_delay = wait_time + 1
                if attempt < retry_attempts - 1:
                    self.logger.info(f"Retrying in {retry_delay:.2f} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.logger.error(
                        f"Exceeded retries for edge {edge.source} -> {edge.target}"
                    )
                    raise e
            except Exception as e:
                self.logger.error(
                    f"Error on edge {edge.source} -> {edge.target}: {e}. Retrying with more depth."
                )
                time.sleep(retry_delay)
                max_depth += 1

        if resp is None:
            raise RuntimeError(
                f"Failed to get a valid response for edge {edge.source} -> {edge.target} after all retries."
            )

        end_time = time.time()
        elapsed_time = end_time - start_time

        self.G.edges._adjdict[edge.source][edge.target]["domain_label"] = (
            resp.domain_label
        )
        self.G.edges._adjdict[edge.source][edge.target]["reasoning"] = (
            resp.reasoning
        )
        self.G_with_context.edges[edge_num_index] = LabelledEdge.from_edge(
            edge, resp.domain_label, resp.reasoning
        )

        sleep_time = max(0, self.config.sleep_interval - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def insert_missing_value_for_edge(self, edge, edge_num_index):
        self.G.edges._adjdict[edge.source][edge.target]["domain_label"] = "MISSING"
        self.G_with_context.edges[edge_num_index] = LabelledEdge.from_edge(
            edge, "MISSING"
        )

    def get_total_tokens_used(self):
        return self.total_tokens_used

    def label_graph(self):
        self.G_with_context.populate_edge_dict()

        if len(self.G_with_context.edges) > 110:
            self.logger.warning(
                f"Graph has {len(self.G_with_context.edges)} edges. This may take a long time and be costly."
            )

        edges_to_process = list(enumerate(self.G_with_context.edges))

        for edge_index, edge in tqdm(
            edges_to_process,
            total=len(edges_to_process),
            desc="Processing graph edges:",
        ):
            try:
                self.label_edge(edge, edge_index, max_depth=self.config.max_depth)
                self.logger.info(
                    f"Successfully labelled edge {edge.source} -> {edge.target}"
                )
            except TokenLimitExceededError as e:
                self.logger.warning(str(e))
                self.logger.info("Stopping graph labeling due to token limit.")
                remaining_start_index = edge_index
                self.logger.info(
                    f"Marking the remaining {len(edges_to_process) - remaining_start_index} edges as 'MISSING'."
                )
                for i in range(remaining_start_index, len(edges_to_process)):
                    unprocessed_edge_index, unprocessed_edge = edges_to_process[i]
                    self.insert_missing_value_for_edge(
                        unprocessed_edge, unprocessed_edge_index
                    )
                break
            except Exception as e:
                self.logger.error(
                    f"Failed to label edge {edge.source} -> {edge.target}: {e}. Marking as 'MISSING'."
                )
                self.insert_missing_value_for_edge(edge, edge_index)

        return self.G, self.G_with_context


if __name__ == "__main__":
    # --- Example Usage ---
    code = """import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'])"""
    from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph

    dfg = DataFlowGraph()
    dfg.parse_code(code)
    dfg.optimize()
    demo_graph_path = os.path.join(os.getcwd(), "demo.execution_graph")
    demo_code_path = os.path.join(os.getcwd(), "demo.code")
    dfg.save_dfg(demo_graph_path)
    with open(demo_code_path, "w") as f:
        f.write(code)

    config = Config("gemini-2.5-flash", 0, 4, llm_provider="google", retry_attempts=2, retry_delay=0, success_delay=0)
    labeler = GraphLabeler(config, demo_graph_path, demo_code_path)
    G, G_with_context = labeler.label_graph()

    output_directory = os.getcwd()
    os.makedirs(output_directory, exist_ok=True)

    output_file = os.path.join(output_directory, "demo.gml")
    import networkx as nx

    nx.write_gml(G, output_file)
    # Clean up

    os.remove(demo_graph_path)
    os.remove(demo_code_path)