import os
import instructor
import time
import logging
import re
from typing import Literal
from tqdm import tqdm
from dotenv import load_dotenv

from ApexDAG.sca.graph_utils import load_graph
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.label_notebooks.message_template import generate_message
from ApexDAG.label_notebooks.pydantic_models import (
    LabelledEdge,
    GraphContextWithSubgraphSearch,
    SubgraphContext,
)

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[logging.FileHandler("graph_labeling.log"), logging.StreamHandler()],
)


class TokenLimitExceededError(Exception):
    """Custom exception raised when the configured token limit is reached."""

    pass


class GraphLabeler:
    """
    Labels graph edges using an LLM, with support for multiple providers and token counting.
    """

    def __init__(self, config: Config, graph_path: str, code_path: str):
        load_dotenv()
        self.config = config
        self.graph_path = graph_path
        self.G = load_graph(self.graph_path)

        self.llm_provider = getattr(config, "llm_provider", "groq")
        self.client = self._initialize_client()

        self.G_with_context = GraphContextWithSubgraphSearch.from_graph(self.G)
        self.code_lines = self.read_code(code_path)

        self.total_tokens_used = 0
        logging.info(
            f"Initialized GraphLabeler with provider '{self.llm_provider}' and model '{self.config.model_name}'."
        )
        if hasattr(self.config, "max_tokens") and self.config.max_tokens > 0:
            logging.info(f"Token limit set to {self.config.max_tokens} tokens.")
        else:
            logging.warning(
                "No max_tokens limit set in config. Running without a token budget."
            )
            self.config.max_tokens = float("inf")

    def _initialize_client(self):
        """
        Initializes and patches the LLM client, importing libraries only when needed.
        """
        logging.info(f"Initializing client for provider: {self.llm_provider}")
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
            return instructor.from_gemini(genai.GenerativeModel(self.config.model_name))

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
            usage = response._raw.usage
            usage_dict = {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
            }
            return response, usage_dict

        elif self.llm_provider == "google":
            response = self.client.create(
                response_model=response_model,
                messages=messages,
            )
            return response, {"prompt": 0, "completion": 0}

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
        retry_attempts = 1
        retry_delay = 1
        success_delay = 1
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

                api_response, usage = self._create_chat_completion(
                    messages=messages,
                    response_model=LabelledEdge,
                )
                resp = api_response

                tokens_this_call = usage["prompt"] + usage["completion"]
                self.total_tokens_used += tokens_this_call
                logging.info(
                    f"Tokens used for ({edge.source} -> {edge.target}): {tokens_this_call}. "
                    f"Total used: {self.total_tokens_used}/{self.config.max_tokens}"
                )

                if resp.domain_label != "MORE_CONTEXT_NEEDED":
                    time.sleep(success_delay)
                    break
                else:
                    logging.info(
                        "'MORE_CONTEXT_NEEDED' received. Retrying with increased depth."
                    )
                    time.sleep(retry_delay)
                    max_depth += 1
            except instructor.exceptions.InstructorRetryException as e:
                logging.error(
                    f"Rate limit error on edge {edge.source} -> {edge.target}: {e}"
                )
                if getattr(e, "status_code", None) == 413:
                    allow_all_code = False
                wait_time = self.extract_wait_time(str(e))
                if wait_time:
                    retry_delay = wait_time + 1
                if attempt < retry_attempts - 1:
                    logging.info(f"Retrying in {retry_delay:.2f} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logging.error(
                        f"Exceeded retries for edge {edge.source} -> {edge.target}"
                    )
                    raise e
            except Exception as e:
                logging.error(
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
        self.G_with_context.edges[edge_num_index] = LabelledEdge.from_edge(
            edge, resp.domain_label
        )

        sleep_time = max(0, self.config.sleep_interval - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def insert_missing_value_for_edge(self, edge, edge_num_index):
        self.G.edges._adjdict[edge.source][edge.target]["domain_label"] = "MISSING"
        self.G_with_context.edges[edge_num_index] = LabelledEdge.from_edge(
            edge, "MISSING"
        )

    def label_graph(self):
        self.G_with_context.populate_edge_dict()

        if len(self.G_with_context.edges) > 110:
            logging.warning(
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
                logging.info(
                    f"Successfully labelled edge {edge.source} -> {edge.target}"
                )
            except TokenLimitExceededError as e:
                logging.warning(str(e))
                logging.info("Stopping graph labeling due to token limit.")
                remaining_start_index = edge_index
                logging.info(
                    f"Marking the remaining {len(edges_to_process) - remaining_start_index} edges as 'MISSING'."
                )
                for i in range(remaining_start_index, len(edges_to_process)):
                    unprocessed_edge_index, unprocessed_edge = edges_to_process[i]
                    self.insert_missing_value_for_edge(
                        unprocessed_edge, unprocessed_edge_index
                    )
                break
            except Exception as e:
                logging.error(
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

    config = Config("gemini-2.5-flash", 2, 2, llm_provider="google")
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
