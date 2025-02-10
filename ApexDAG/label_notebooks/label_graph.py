

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
from ApexDAG.label_notebooks.pydantic_models import LabelledEdge, GraphContextWithSubgraphSearch, SubgraphContext
from ApexDAG.label_notebooks.utils import get_input_subgraph, load_config, Config


def label_edge(G_with_context, edge, edge_num_index, client, config, max_depth=2):
    
    start_time = time.time()
    
    try:
        graph_context = get_input_subgraph(G_with_context, edge.source, edge.target, max_depth=max_depth)
        
        resp = client.chat.completions.create(
            model=config.model_name,
            messages = generate_message(edge.source, edge.target, edge.code, graph_context),
            response_model=LabelledEdge,
        )
    except Exception as _:
        graph_context = get_input_subgraph(G_with_context, edge.source, edge.target, max_depth=max_depth + 1)
        
        resp = client.chat.completions.create(
            model=config.model_name,
            messages = generate_message(edge.source, edge.target, edge.code, graph_context),
            response_model=LabelledEdge,
        )
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    G.edges._adjdict[edge.source][edge.target]["domain_label"] = resp.domain_label # serialize into labelled node! 
    G_with_context.edges[edge_num_index] = LabelledEdge.from_edge(edge, resp.domain_label)
        
    sleep_time = max(0, config.sleep_interval - elapsed_time)
        
    if sleep_time > 0:
        time.sleep(sleep_time)  # sleep to comply with the rate limit
    return G, G_with_context

def insert_missing_value_for_edge(G_with_context, edge, edge_num_index):
    G.edges._adjdict[edge.source][edge.target]["domain_label"] = 'MISSING' # serialize into labelled node! 
    G_with_context.nodes[edge_num_index] = LabelledEdge.from_edge(edge, 'MISSING')
    return G, G_with_context
            
def label_graph(config: Config, G: nx.DiGraph):
    G_with_context = GraphContextWithSubgraphSearch.from_graph(G)
    G_with_context.populate_edge_dict()
    
    client = Groq(
        api_key=os.environ.get('GROQ_API_KEY'),
    )
    client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

    for edge_index, edge in tqdm(enumerate(G_with_context.edges), total=len(G_with_context.edges), desc=f"Processing nodes of graph {args.graph_path}"):
        try:
            G, G_with_context = label_edge(G_with_context, edge, edge_index, client, config)
        except Exception as e:
            print(f"Error: {e}") # TODO: propper logging exception, but there is no propper logging yet
            G, G_with_context = insert_missing_value_for_edge(G_with_context, edge, edge_index)
            
        
    return G, G_with_context
        
    
if __name__ == "__main__":
    # example of single use
    args = argparse.ArgumentParser()
    
    args.add_argument("--config_path", type=str, default= "ApexDAG/label_notebooks/config.yaml", help="Name of the model to use for labeling")
    args.add_argument("--graph_path", type=str, default= "output/graph.gml", help="Path to the graph file")
    args = args.parse_args()
    
    config = load_config(args.config_path)
    
    G = load_graph(args.graph_path)
    G, G_with_context = label_graph(config, G)
    
    for source in G.edges._adjdict:
        for target in G.edges._adjdict[source]:
            edge_index = G_with_context.edge_dict[source][target]
            G.edges._adjdict[source][target]["domain_label"] = G_with_context.edges[edge_index].domain_label
    
        
    directory_path = f"output/labelling/labelled_graph_{config.model_name}_data/"
    os.makedirs(directory_path, exist_ok=True)
    nx.write_gml(G, f"{directory_path}/labelled.gml")

    # read the saved graph 
    G = nx.read_gml(f"{directory_path}/labelled.gml")
    drawer = Draw(NODE_TYPES, EDGE_TYPES)
    drawer.labelled_dfg(G, f'{directory_path}labelled_dfg') 