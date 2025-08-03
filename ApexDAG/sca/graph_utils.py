import os
import networkx as nx
from ApexDAG.util.draw import Draw
from ApexDAG.util.logging import setup_logging


def convert_multidigraph_to_digraph(G: nx.MultiDiGraph, node_types: dict, verbose=False) -> nx.DiGraph:
    logger = setup_logging("graph_utils", verbose=verbose)
    G = G.copy()
    new_G = nx.DiGraph()

    errors = []
    # Iterate through the nodes to copy their attributes
    for node, attrs in G.nodes(data=True):
        if "label" not in attrs:
            errors.append(f"Node {node} is missing attribute(s): {str(attrs)}")
        else:
            new_G.add_node(node, label=attrs["label"], node_type=attrs["node_type"])

    if len(errors) > 0:
        raise AttributeError(", ".join(errors))

    # Track processed edges to avoid duplication
    processed_edges = set()

    # Iterate through the edges in the original graph
    for u, v, data in G.edges(data=True):
        edge_pair = (u, v)

        if G.number_of_edges(u, v) > 1 and edge_pair not in processed_edges:
            logger.debug("Processing multiple edges between %s and %s", str(u), str(v))
            # Process all edges between u and v
            edges = list(G.get_edge_data(u, v).items())

            # Create intermediate nodes and edges for multiple edges
            for i, (_, edge_data) in enumerate(edges):
                if i == 0:
                    # Create the first intermediate node and edge
                    intermediate_node = f"{v}_intermediate_1"
                    new_G.add_node(intermediate_node, label=intermediate_node, node_type=node_types["INTERMEDIATE"])
                    if "predicted_label" in edge_data:
                        new_G.add_edge(u, intermediate_node, code=edge_data["code"], edge_type=edge_data["edge_type"], predicted_label=edge_data["predicted_label"])
                    else:
                        new_G.add_edge(u, intermediate_node, code=edge_data["code"], edge_type=edge_data["edge_type"])
                elif i < len(edges) - 1:
                    # Create subsequent intermediate nodes and edges
                    intermediate_node_prev = f"{v}_intermediate_{i}"
                    intermediate_node = f"{v}_intermediate_{i+1}"
                    new_G.add_node(intermediate_node, label=intermediate_node, node_type=node_types["INTERMEDIATE"])
                    if "predicted_label" in edge_data:
                        new_G.add_edge(intermediate_node_prev, intermediate_node, code=edge_data["code"], edge_type=edge_data["edge_type"], predicted_label=edge_data["predicted_label"])
                    else:
                        new_G.add_edge(intermediate_node_prev, intermediate_node, code=edge_data["code"], edge_type=edge_data["edge_type"])
                else:
                    # Connect the last intermediate node to the original destination node
                    intermediate_node_prev = f"{v}_intermediate_{i}"
                    if "predicted_label" in edge_data:
                        new_G.add_edge(intermediate_node_prev, v, code=edge_data["code"], edge_type=edge_data["edge_type"], predicted_label=edge_data["predicted_label"])
                    else:
                        new_G.add_edge(intermediate_node_prev, v, code=edge_data["code"], edge_type=edge_data["edge_type"])

            # Mark this pair as processed
            processed_edges.add(edge_pair)

        elif edge_pair not in processed_edges:
            # If there's only one edge, copy it directly
            new_G.add_edge(u, v, **data)
            processed_edges.add(edge_pair)

    return new_G

def get_subgraph(G: nx.DiGraph, variable_versions: dict, variable: str) -> nx.DiGraph:
    if variable not in variable_versions:
        raise ValueError(f"Variable {variable} not found in the graph")

    # get first variable node
    variable = variable_versions[variable][0]
    G_copy = G.copy()
    ancestors = nx.ancestors(G_copy, variable)
    descendants = nx.descendants(G_copy, variable)
    relevant_nodes = descendants.union(ancestors)
    relevant_nodes.add(variable)
    
    return G_copy.subgraph(relevant_nodes).copy()

def get_all_subgraphs(G: nx.DiGraph, variable_versions: dict) -> list[nx.DiGraph]:
    subgraphs = []
    for variable in variable_versions:
        subgraphs.append(get_subgraph(G, variable_versions, variable))
    return subgraphs

def debug_graph(G: nx.DiGraph, prev_graph_path: str, new_graph_path: str, node_types: dict, edge_types: dict, save_prev=False, verbose=False):
    logger = setup_logging("graph_utils", verbose=verbose)

    if os.path.exists(prev_graph_path):
        prev_G = load_graph(prev_graph_path)
    elif save_prev:
        save_graph(G, new_graph_path)
        return
    else:
        return

    edge_types["added"] = -1
    edge_types["modified"] = -2
    edge_types["deleted"] = -3
    node_types["added"] = -1
    node_types["modified"] = -2
    node_types["deleted"] = -3
    
    # Calculate differences between graphs
    added_edges = set(G.edges) - set(prev_G.edges)    
    removed_edges = set(prev_G.edges) - set(G.edges)
    modified_edges = {edge for edge in set(G.edges).intersection(set(prev_G.edges)) if G.get_edge_data(*edge) != prev_G.get_edge_data(*edge)}
    
    added_nodes = set(G.nodes) - set(prev_G.nodes)
    removed_nodes = set(prev_G.nodes) - set(G.nodes)
    modified_nodes = {node for node in set(G.nodes).intersection(set(prev_G.nodes)) if G.nodes[node]["node_type"] != prev_G.nodes[node]["node_type"]}
    
    # Highlight changes in the previous graph
    for u, v, key in added_edges:
        prev_G.add_edge(u, v, key, edge_type=-1, code=f"added -> {G[u][v][key]['code']}")
    
    for u, v, key in modified_edges:
        prev_G[u][v][key]['edge_type'] = -2
        prev_G[u][v][key]['code'] = f"modified -> {G[u][v][key]['code']}"
    
    for u, v, key in removed_edges:
        prev_G[u][v][key]['edge_type'] = -3
        prev_G[u][v][key]['code'] = f"deleted -> {prev_G[u][v][key]['code']}"
    
    for node in added_nodes:
        prev_G.add_node(node, label=G.nodes[node]['label'], node_type=-1)
    
    for node in modified_nodes:
        prev_G.nodes[node]['node_type'] = -2

    for node in removed_nodes:
        prev_G.nodes[node]['node_type'] = -3

    logger.debug(f"Added edges: {added_edges}")
    logger.debug(f"Removed edges: {removed_edges}")
    logger.debug(f"Modified edges: {modified_edges}")
    logger.debug(f"Added nodes: {added_nodes}")
    logger.debug(f"Removed nodes: {removed_nodes}")
    logger.debug(f"Modified nodes: {modified_nodes}")

    if save_prev: 
        logger.debug("Saving debug graph")
        Draw(node_types, edge_types).dfg(prev_G, "debug_graph")
        save_graph(G, new_graph_path)    

def save_graph(G: nx.DiGraph, path: str) -> None:
    nx.write_gml(G, os.path.join(os.getcwd(), path))

def load_graph(path: str) -> nx.DiGraph:
    """
    Load a graph from a saved GraphML file.

    Args:
        path (str): Path to the GraphML file.

    Returns:
        nx.DiGraph: The loaded directed graph.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the graph cannot be loaded due to missing or invalid attributes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    try:
        G = nx.read_gml(path)
        for node, attrs in G.nodes(data=True):
            if 'label' not in attrs:
                G.nodes[node]['label'] = node
            if 'node_type' not in attrs:
                raise ValueError(f"Node {node} is missing required attribute 'node_type'")

        for u, v, data in G.edges(data=True):
            if 'code' not in data or 'edge_type' not in data:
                raise ValueError(f"Edge {u} -> {v} is missing required attributes 'code' or 'edge_type")

        return G
    except Exception as e:
        raise ValueError(f"Failed to load the graph from {path}: {e}")
