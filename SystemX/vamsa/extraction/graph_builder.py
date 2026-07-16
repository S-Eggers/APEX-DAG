import ast

import networkx as nx

from ..core.types import PRType, SpatialMetadata, WIRNode
from ..core.utils import merge_prs, remove_id, reset_vamsa_counter
from .ast_parser import gen_pr
from .filters import filter_PRs, fix_bibartie_issue_import_from, remove_assignments

def construct_bipartite_graph(prs: list[PRType], spatial_registry: dict[str, SpatialMetadata]) -> tuple[nx.DiGraph, tuple[set, set, set, set]]:

    G = nx.DiGraph()
    input_nodes, operation_nodes, caller_nodes, output_nodes = set(), set(), set(), set()

    for input_i, c, p, output_o in prs:
        input_nodes.update([e for e in (input_i,) if e is not None])
        operation_nodes.add(p)
        if c is not None:
            caller_nodes.add(c)
        if output_o is not None:
            output_nodes.add(output_o)

        for input_node in [e for e in (input_i,) if e is not None]:
            G.add_edge(input_node, p, edge_type=1, label="input")
        if c is not None:
            G.add_edge(c, p, edge_type=0, label="caller")
        if output_o is not None:
            G.add_edge(p, output_o, edge_type=2, label="output")

    for node in input_nodes | caller_nodes | output_nodes:
        node_label = remove_id(node)

        if not G.has_node(node):
            G.add_node(node)

        G.nodes[node]["node_type"] = 0
        G.nodes[node]["label"] = node_label

        if str(node) in spatial_registry:
            coords = spatial_registry[str(node)]
            G.nodes[node].update({"lineno": coords.lineno, "col_offset": coords.col_offset, "end_lineno": coords.end_lineno, "end_col_offset": coords.end_col_offset})

    for node in operation_nodes:
        node_label = remove_id(node)
        if not G.has_node(node):
            G.add_node(node)

        G.nodes[node]["node_type"] = 3
        G.nodes[node]["label"] = node_label

        if str(node) in spatial_registry:
            coords = spatial_registry[str(node)]
            G.nodes[node].update({"lineno": coords.lineno, "col_offset": coords.col_offset, "end_lineno": coords.end_lineno, "end_col_offset": coords.end_col_offset})

    return G, (input_nodes, output_nodes, caller_nodes, operation_nodes)

def gen_wir(root: ast.AST) -> tuple[nx.DiGraph, list[PRType], tuple[set, set, set, set]]:
    """The main execution pipeline for the Extraction module."""
    reset_vamsa_counter()
    PRs: list[PRType] = []

    spatial_registry: dict[str, SpatialMetadata] = {}

    for child in ast.iter_child_nodes(root):
        _, PRs_prime = gen_pr(WIRNode(child), PRs, spatial_registry)
        PRs = list(merge_prs(PRs, PRs_prime))

    PRs = [pr for pr in PRs if pr[2] is not None]
    PRs_filtered = filter_PRs(PRs)
    PRs_filtered = fix_bibartie_issue_import_from(PRs_filtered)
    PR_filtered_no_assign = remove_assignments(PRs_filtered)

    G, tuples = construct_bipartite_graph(PR_filtered_no_assign, spatial_registry)

    return G, PR_filtered_no_assign, tuples
