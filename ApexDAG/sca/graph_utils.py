import logging
import os

import networkx as nx

from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


def convert_multidigraph_to_digraph(graph: nx.MultiDiGraph, node_types: dict, verbose: bool = False) -> nx.DiGraph:
    graph = graph.copy()
    new_graph = nx.DiGraph()

    errors = []
    for node, attrs in graph.nodes(data=True):
        if "label" not in attrs:
            errors.append(f"Node {node} is missing attribute(s): {attrs!s}")
        else:
            new_graph.add_node(node, **attrs)

    if len(errors) > 0:
        raise AttributeError(", ".join(errors))

    processed_edges = set()

    for u, v, data in graph.edges(data=True):
        edge_pair = (u, v)

        if graph.number_of_edges(u, v) > 1 and edge_pair not in processed_edges:
            logger.debug("Processing multiple edges between %s and %s", str(u), str(v))
            edges = list(graph.get_edge_data(u, v).items())

            u_node_type = new_graph.nodes[u].get("node_type")
            v_node_type = new_graph.nodes[v].get("node_type")
            intermediate_node_type = node_types["DATASET"] if u_node_type == node_types["DATASET"] and v_node_type == node_types["DATASET"] else node_types["INTERMEDIATE"]

            current_node = u
            for i, (_, edge_data) in enumerate(edges):
                if i < len(edges) - 1:
                    intermediate_node = f"{v}_intermediate_{i + 1}"
                    inherited_cell_id = edge_data.get("cell_id", "unknown_cell")

                    new_graph.add_node(
                        intermediate_node,
                        label=intermediate_node,
                        node_type=intermediate_node_type,
                        cell_id=inherited_cell_id,
                    )
                    target_node = intermediate_node
                else:
                    target_node = v

                edge_attrs = {k: val for k, val in edge_data.items() if k != "key"}

                new_graph.add_edge(current_node, target_node, **edge_attrs)
                current_node = target_node

            processed_edges.add(edge_pair)

        elif edge_pair not in processed_edges:
            edge_attrs = {k: val for k, val in data.items() if k != "key"}
            new_graph.add_edge(u, v, **edge_attrs)
            processed_edges.add(edge_pair)

    return new_graph


def get_subgraph(graph: nx.DiGraph, variable_versions: dict, variable: str) -> nx.DiGraph:
    if variable not in variable_versions:
        raise ValueError(f"Variable {variable} not found in the graph")

    # get first variable node
    variable = variable_versions[variable][0]
    graph_copy = graph.copy()
    ancestors = nx.ancestors(graph_copy, variable)
    descendants = nx.descendants(graph_copy, variable)
    relevant_nodes = descendants.union(ancestors)
    relevant_nodes.add(variable)

    return graph_copy.subgraph(relevant_nodes).copy()


def get_all_subgraphs(graph: nx.DiGraph, variable_versions: dict) -> list[nx.DiGraph]:
    subgraphs = []
    for variable in variable_versions:
        subgraphs.append(get_subgraph(graph, variable_versions, variable))
    return subgraphs


def save_graph(graph: nx.DiGraph, path: str) -> None:
    nx.write_gml(graph, os.path.join(os.getcwd(), path))


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
        graph = nx.read_gml(path)
        for node, attrs in graph.nodes(data=True):
            if "label" not in attrs:
                graph.nodes[node]["label"] = node
            if "node_type" not in attrs:
                raise ValueError(f"Node {node} is missing required attribute 'node_type'")

        for u, v, data in graph.edges(data=True):
            if "code" not in data or "edge_type" not in data:
                raise ValueError(f"Edge {u} -> {v} is missing required attributes 'code' or 'edge_type")

        return graph
    except Exception as e:
        raise ValueError(f"Failed to load the graph from {path}: {e}") from None
