import instructor
import os
import argparse
import networkx as nx

from groq import Groq
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

    for node in G_with_context.nodes:
        subgraph_nodes, subgraph_edges = G_with_context.get_subgraph(node.id)
        subgraph_with_context = SubgraphContext(
            node_of_interest=node.id,
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges
        )
        
        graph_context = get_input_subgraph(subgraph_with_context, node.id)
        resp = client.chat.completions.create(
            model=config.model_name,
            messages = generate_message(node,graph_context),
            response_model=LabelledNode,
        )
        G.nodes[node]["domainlabel"] = resp.domain_label # serialize into labelled node! 
        G_with_context.nodes[node.id] = LabelledNode.from_node(node, resp.domain_label)
        
    
if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, default= "ApexDAG\label_notebooks\config.yaml", help="Name of the model to use for labeling")
    args.add_argument("--graph_path", type=str, default= "output/tests-labeling/bhuvankumaru_DWR_notebooks_ARIA_vs_TRE_timeseries_comparison_ipynb.execution_graph", help="Path to the graph file")
    args = args.parse_args()
    
    config = load_config(args.config_path)
    G = load_graph(args.graph_path)
    
    G_with_context = GraphContextWithSubgraphSearch.from_graph(G)
    
    client = Groq(
        api_key=os.environ.get('GROQ_API_KEY'),
    )
    client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

    for node in G_with_context.nodes:
        subgraph_nodes, subgraph_edges = G_with_context.get_subgraph(node.id)
        graph_context = get_input_subgraph(G_with_context, node.id)
        resp = client.chat.completions.create(
            model=config.model_name,
            messages = generate_message(node,graph_context),
            response_model=LabelledNode,
        )
        G.nodes[node]["domainlabel"] = resp.domain_label # serialize into labelled node! 
        G_with_context.nodes[node.id] = LabelledNode.from_node(node, resp.domain_labell)
        
    # now make domainlabel to normal label
    for node in G.nodes:
        G.nodes[node]["label"] = f"{G.nodes[node]["domainlabel"]}: {G.nodes[node]["domainlabel"]}"
        
    # draw it
    drawer = Draw(NODE_TYPES, EDGE_TYPES)
    drawer.dfg(G, "labelled_graph")  # Save the drawn graph as an image
        
        
