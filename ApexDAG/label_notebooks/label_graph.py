

import instructor
import os
import time
import argparse
import networkx as nx

from groq import Groq
from tqdm import tqdm
from ApexDAG.util.draw import Draw
from ApexDAG.sca.constants import NODE_TYPES, EDGE_TYPES
from ApexDAG.sca.graph_utils import load_graph
from ApexDAG.label_notebooks.message_template import generate_message
from ApexDAG.label_notebooks.pydantic_models import LabelledNode, GraphContextWithSubgraphSearch, SubgraphContext
from ApexDAG.label_notebooks.utils import get_input_subgraph, load_config, Config


def label_graph(config: Config, G: nx.DiGraph):
    G_with_context = GraphContextWithSubgraphSearch.from_graph(G)
    
    client = Groq(
        api_key=os.environ.get('GROQ_API_KEY'),
    )
    client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

    for node_num_index, node in tqdm(enumerate(G_with_context.nodes), total=len(G_with_context.nodes), desc=f"Processing nodes of graph {args.graph_path}"):
        start_time = time.time()
        subgraph_nodes, subgraph_edges = G_with_context.get_subgraph(node.id)
        subgraph_with_context = SubgraphContext(
            node_of_interest=node.id,
            nodes=subgraph_nodes,
            edges=subgraph_edges
        )
        graph_context = get_input_subgraph(subgraph_with_context, node.id)
        
        resp = client.chat.completions.create(
            model=config.model_name,
            messages = generate_message(node, graph_context),
            response_model=LabelledNode,
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        G.nodes._nodes[node.id]["domain_label"] = resp.domain_label # serialize into labelled node! 
        G_with_context.nodes[node_num_index] = LabelledNode.from_node(node, resp.domain_label)
        
        sleep_time = max(0, config.sleep_interval - elapsed_time)
        
        if sleep_time > 0:
            time.sleep(sleep_time)  # sleep to comply with the rate limit
            
        
    return G, G_with_context
        
    
if __name__ == "__main__":

    args = argparse.ArgumentParser()
    
    args.add_argument("--config_path", type=str, default= "ApexDAG/label_notebooks\config.yaml", help="Name of the model to use for labeling")
    args.add_argument("--graph_path", type=str, default= "output/tests-labeling/bhuvankumaru_DWR_notebooks_ARIA_vs_TRE_timeseries_comparison_ipynb.execution_graph", help="Path to the graph file")
    
    args = args.parse_args()
    
    config = load_config(args.config_path)
    G = load_graph(args.graph_path)
    
    G, G_with_context = label_graph(config, G)
    
    for node in G.nodes:
        G.nodes[node]["label"] = f"{G.nodes[node]['domain_label']}: {G.nodes[node]['domain_label']}"
        
    drawer = Draw(NODE_TYPES, EDGE_TYPES)
    drawer.dfg(G, "labelled_graph_{}")  # Save the drawn graph as an image
