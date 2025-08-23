import os
import re
import json
import time
import logging
import instructor
import networkx as nx
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
import concurrent.futures
from pydantic import TypeAdapter, ValidationError

from ApexDAG.sca.graph_utils import load_graph
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.label_notebooks.message_template import generate_message
from ApexDAG.label_notebooks.pydantic_models_multigraph import (
    MultiLabelledEdge,
    MultiGraphContextWithSubgraphSearch,
    MultiSubgraphContext,
)

log_format = "{asctime} - {name} - {levelname} - {message}"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    style="{",
    handlers=[logging.FileHandler("graph_labeling.log"), logging.StreamHandler()],
)


class OnlineGraphLabeler:
    """
    Labels graph edges by making concurrent requests to an LLM using a thread pool.

    This version is optimized for speed and robustness:
    1.  Uses a ThreadPoolExecutor to make many API calls simultaneously.
    2.  Each edge labeling is an independent, "one-shot" attempt.
    3.  Does not track token usage.
    """
    def __init__(self, config: Config, graph: nx.MultiDiGraph, code: str):
        load_dotenv()
        self.config = config
        self.G = graph

        self.llm_provider = getattr(config, "llm_provider", "groq")
        self.client = self._initialize_client()

        self.G_with_context = MultiGraphContextWithSubgraphSearch.from_graph(self.G)
        self.code_lines = code.splitlines()

        logging.info(
            f"Initialized GraphLabeler with provider '{self.llm_provider}', model '{self.config.model_name}', "
            f"and max workers {self.config.max_workers}."
        )

    def _initialize_client(self):
        """Initializes and patches the LLM client based on the provider."""
        logging.info(f"Initializing client for provider: {self.llm_provider}")
        if self.llm_provider == "groq":
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            client = Groq(api_key=api_key)
            return instructor.from_groq(client, mode=instructor.Mode.TOOLS)

        elif self.llm_provider == "google":
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError("google-generativeai is not installed.") from e
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(self.config.model_name)
            return client
        else:
            raise ValueError(f"Unsupported LLM provider: '{self.llm_provider}'.")

    def _create_chat_completion(self, messages, response_model):
        """Makes a generic API call to the configured LLM."""
        if self.llm_provider == "groq":
            return self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                response_model=response_model,
            )
        elif self.llm_provider == "google":
            gemini_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
            response = self.client.generate_content(
                gemini_messages,
                generation_config={"response_mime_type": "application/json"},
            )
            try:
                parsed_response = json.loads(response.text)
                return TypeAdapter(response_model).validate_python(parsed_response)
            except (json.JSONDecodeError, ValidationError) as e:
                logging.error(f"Error parsing/validating Gemini response: {e}")
                logging.error(f"Raw Gemini response: {response.text}")
                raise

    def get_input_subgraph(self, node_id_source: str, node_id_target: str, edge_key: str, max_depth: int):
        """Extracts a subgraph around an edge of interest."""
        subgraph_nodes, subgraph_edges = self.G_with_context.get_subgraph(
            node_id_source, node_id_target, max_depth=max_depth
        )
        return str(MultiSubgraphContext(
            edge_of_interest=(node_id_source, node_id_target, edge_key),
            nodes=subgraph_nodes,
            edges=subgraph_edges,
        ))

    def get_input_code_context(self, node_id_source: str, node_id_target: str, max_depth: int):
        """Extracts relevant lines of code for the subgraph."""
        _, subgraph_edges = self.G_with_context.get_subgraph(
            node_id_source, node_id_target, max_depth=max_depth
        )
        lines = sorted({line for edge in subgraph_edges for line in (edge.lineno or [])})
        if -1 in lines:
            return "\n".join(self.code_lines)
        
        expanded_lines = sorted({
            line_offset
            for line in lines
            for line_offset in (line - 1, line, line + 1)
            if 0 <= line_offset < len(self.code_lines)
        })
        return "\n".join(self.code_lines[line] for line in expanded_lines)

    def _label_edge_worker(self, edge, edge_index, max_depth):
        """
        Worker function to label a single edge.
        It is designed to be thread-safe by returning results instead of modifying the graph directly.
        """
        start_time = time.time()
        
        for attempt in range(self.config.retry_attempts):
            try:
                graph_context = self.get_input_subgraph(
                    edge.source, edge.target, edge.key, max_depth
                )
                code_context = self.get_input_code_context(
                    edge.source, edge.target, max_depth
                )
                messages = generate_message(
                    edge.source, edge.target, edge.key, edge.code, graph_context, code_context
                )

                response = self._create_chat_completion(
                    messages=messages,
                    response_model=MultiLabelledEdge,
                )

                return {
                    "edge_index": edge_index,
                    "original_edge": edge,
                    "labeled_edge": response,
                }
            except Exception as e:
                logging.warning(
                    f"Attempt {attempt + 1} failed for edge {edge.source}->{edge.target}: {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise e
        
        raise RuntimeError(f"Failed to label edge {edge.source}->{edge.target} after all retries.")


    def label_graph(self):
        """Labels all edges in the graph concurrently using a thread pool."""
        self.G_with_context.populate_edge_dict()
        edges_to_process = list(enumerate(self.G_with_context.edges))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_edge_info = {
                executor.submit(
                    self._label_edge_worker, edge, edge_index, self.config.max_depth
                ): (edge_index, edge)
                for edge_index, edge in edges_to_process
            }

            pbar = tqdm(
                concurrent.futures.as_completed(future_to_edge_info),
                total=len(edges_to_process),
                desc="Labeling edges concurrently",
            )
            
            for future in pbar:
                edge_index, edge = future_to_edge_info[future]
                try:
                    # Get the result from the completed future
                    result = future.result()
                    labeled_edge = result["labeled_edge"]

                    self.G.edges[edge.source, edge.target, edge.key]["domain_label"] = labeled_edge.domain_label
                    self.G.edges[edge.source, edge.target, edge.key]["reasoning"] = labeled_edge.reasoning
                    self.G_with_context.edges[edge_index] = labeled_edge

                except Exception as e:
                    logging.error(
                        f"Failed to label edge {edge.source}->{edge.target} after all retries: {e}. Marking as 'MISSING'."
                    )
                    self.insert_missing_value_for_edge(edge, edge_index)

        return self.G, self.G_with_context

    def insert_missing_value_for_edge(self, edge, edge_num_index):
        """Helper to mark an edge with a 'MISSING' label."""
        self.G.edges[edge.source, edge.target, edge.key]["domain_label"] = "MISSING"
        self.G_with_context.edges[edge_num_index] = MultiLabelledEdge.from_edge(edge, "MISSING", "Failed during processing.")


if __name__ == "__main__":
    from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph

    # --- Example Usage ---
    code = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("data.csv")
X = df.drop('target', axis=1)
y = df['target']

# Feature Engineering: Create a new feature
X['new_feature'] = X['feature1'] * X['feature2']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")"""

    dfg = DataFlowGraph()
    dfg.parse_code(code)
    dfg.optimize()

    config = Config(
        model_name="gemini-1.5-flash",
        sleep_interval=0,
        max_depth=2,
        llm_provider="google",
        retry_attempts=2,
        retry_delay=1,
        max_workers=10,
    )
    
    labeler = OnlineGraphLabeler(config, dfg.get_graph(), dfg.code)
    G, G_with_context = labeler.label_graph()

    # Print the results
    attrs_to_set = {}
    for u, v, key, data in G.edges(data=True, keys=True):
        if "domain_label" in data:
            attrs_to_set[f"{u}->{v} ({key})"] = data["domain_label"]
    
    print("\n--- Labeled Edges ---")
    print(json.dumps(attrs_to_set, indent=2))