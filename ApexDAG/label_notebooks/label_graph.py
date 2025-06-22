import os
import instructor
import time
import logging
import re

from groq import Groq
from tqdm import tqdm
from ApexDAG.sca.graph_utils import load_graph
from ApexDAG.label_notebooks.utils import Config
from ApexDAG.label_notebooks.message_template import generate_message
from ApexDAG.label_notebooks.pydantic_models import LabelledEdge, GraphContextWithSubgraphSearch
from ApexDAG.label_notebooks.pydantic_models import GraphContextWithSubgraphSearch, SubgraphContext


log_format = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("graph_labeling.log"),
        logging.StreamHandler()
    ]
)
class GraphLabeler:
    def __init__(self, config: Config, graph_path: str, code_path: str):
        self.config = config
        self.graph_path = graph_path
        
        self.G = load_graph(self.graph_path)
        self.client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        self.client = instructor.from_groq(self.client, mode=instructor.Mode.TOOLS)
        
        self.G_with_context = GraphContextWithSubgraphSearch.from_graph(self.G)
        self.code_lines = self.read_code(code_path)
        
        
    def read_code(self, code_path: str):
        if not os.path.exists(code_path):
            raise FileNotFoundError(f"The file at '{code_path}' does not exist.")
        with open(code_path, 'r') as f:
            code = f.read()
        return code.splitlines()
        
    def get_input_subgraph(self, node_id_source: str, node_id_target: str, max_depth: int = 1):
        subgraph_nodes, subgraph_edges = self.G_with_context.get_subgraph(node_id_source, node_id_target, max_depth = max_depth)
        model_input = SubgraphContext(
            edge_of_interest=(node_id_source, node_id_target),
            nodes=subgraph_nodes,
            edges=subgraph_edges
        )
        input_graph_structure = str(model_input)
        return input_graph_structure 
    
    def get_input_code_context(self, node_id_source: str, node_id_target: str, max_depth: int = 1, allow_all_code: bool = True):
        _, subgraph_edges = self.G_with_context.get_subgraph(node_id_source, node_id_target, max_depth=max_depth)

        lines = sorted({
            line
            for edge in subgraph_edges
            for line in (edge.lineno or [])
        })

        if (-1 in lines) and allow_all_code:
            return "\n".join(self.code_lines)
        
        expanded_lines = sorted({line_offset for line in lines for line_offset in (line - 1, line, line + 1) if 0 <= line_offset < len(self.code_lines)})
        return "\n".join(self.code_lines[line] for line in expanded_lines)
                

    def extract_wait_time(self, error_message):
        match = re.search(r'Please try again in (\d+)m(\d+\.\d+)s', error_message)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        return None

    def label_edge(self, edge, edge_num_index, max_depth=2):
        start_time = time.time()
        retry_attempts = 3
        retry_delay = 60  # initial delay in seconds
        success_delay = 10
        allow_all_code = True 
        
        for attempt in range(retry_attempts):
            try:
                graph_context = self.get_input_subgraph(edge.source, edge.target, max_depth=max_depth)
                code_context = self.get_input_code_context(edge.source, edge.target, max_depth=max_depth, allow_all_code=allow_all_code)
                
                resp = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=generate_message(edge.source, edge.target, edge.code, graph_context, code_context),
                    response_model=LabelledEdge,
                )
                if resp.domain_label != "MORE_CONTEXT_NEEDED":
                    time.sleep(success_delay)
                    break
                else:
                    time.sleep(retry_delay)
                    max_depth += 1
            except instructor.exceptions.InstructorRetryException as e:
                logging.error(f"Rate limit error during request for edge {edge.source} -> {edge.target}: {e}")
                # if error code is 413 set allow all code to False
                if getattr(e, 'status_code', None) == 413:
                    allow_all_code = False
                wait_time = self.extract_wait_time(str(e))
                if wait_time:
                    retry_delay = wait_time + 1
                if attempt < retry_attempts - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    logging.error(f"Exceeded maximum retry attempts for edge {edge.source} -> {edge.target}")
                    raise e
            except Exception as e:
                logging.error(f"Error during initial request for edge {edge.source} -> {edge.target}: {e}. Trying with depth + 1.")
                time.sleep(retry_delay)
                max_depth += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        self.G.edges._adjdict[edge.source][edge.target]["domain_label"] = resp.domain_label 
        self.G_with_context.edges[edge_num_index] = LabelledEdge.from_edge(edge, resp.domain_label)
            
        sleep_time = max(0, self.config.sleep_interval - elapsed_time)
            
        if sleep_time > 0:
            time.sleep(sleep_time)  # sleep to comply with the rate limit

    def insert_missing_value_for_edge(self, edge, edge_num_index):
        self.G.edges._adjdict[edge.source][edge.target]["domain_label"] = 'MISSING' 
        self.G_with_context.edges[edge_num_index] = LabelledEdge.from_edge(edge, 'MISSING')

    def label_graph(self):
        
        self.G_with_context.populate_edge_dict()

        if len(self.G_with_context.edges) > 110:
            logging.warning(f"Graph has more than 100 edges ({len(self.G_with_context.edges)}). This may take a long time to process.")
            raise ValueError("Graph has more than 100 edges. Please reduce the number of edges before processing. Skipping")
        
        for edge_index, edge in tqdm(enumerate(self.G_with_context.edges), total=len(self.G_with_context.edges), desc="Processing nodes of graph:"):
            try:
                self.label_edge(edge, edge_index, max_depth = self.config.max_depth)
                logging.info(f"Successfully labelled edge {edge.source} -> {edge.target}")
            except Exception as e:
                logging.error(f"Error during edge labelling: {e}. Filling value with missing.")
                self.insert_missing_value_for_edge(edge, edge_index)
        
        return self.G, self.G_with_context
