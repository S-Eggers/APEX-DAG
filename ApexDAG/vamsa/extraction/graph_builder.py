import ast

import networkx as nx

from ..core.types import PRType, WIRNode
from ..core.utils import merge_prs, remove_id, reset_vamsa_counter
from .ast_parser import GenPR
from .filters import filter_PRs, fix_bibartie_issue_import_from, remove_assignments


def construct_bipartite_graph(PRs: list[PRType]) -> tuple[nx.DiGraph, tuple[set, set, set, set]]:
    G = nx.DiGraph()
    input_nodes, operation_nodes, caller_nodes, output_nodes = set(), set(), set(), set()

    for I, c, p, O in PRs:
        input_nodes.update([e for e in (I,) if e is not None])
        operation_nodes.add(p)
        if c is not None: caller_nodes.add(c)
        if O is not None: output_nodes.add(O)

        for input_node in [e for e in (I,) if e is not None]:
            G.add_edge(input_node, p, edge_type=1, label="input")
        if c is not None:
            G.add_edge(c, p, edge_type=0, label="caller")
        if O is not None:
            G.add_edge(p, O, edge_type=2, label="output")

    for node in input_nodes | caller_nodes | output_nodes:
        if not G.has_node(node): G.add_node(node)
        G.nodes[node]["node_type"] = 0
        G.nodes[node]["label"] = remove_id(node)

    for node in operation_nodes:
        if not G.has_node(node): G.add_node(node)
        G.nodes[node]["node_type"] = 3
        G.nodes[node]["label"] = remove_id(node)

    return G, (input_nodes, output_nodes, caller_nodes, operation_nodes)

def GenWIR(root: ast.AST) -> tuple[nx.DiGraph, list[PRType], tuple[set, set, set, set]]:
    """
    The main execution pipeline for the Extraction module.
    Parses the AST, filters PRs, and outputs the graph.
    """
    reset_vamsa_counter()
    PRs: list[PRType] = []

    # 1. AST Traversal
    for child in ast.iter_child_nodes(root):
        _, PRs_prime = GenPR(WIRNode(child), PRs)
        PRs = list(merge_prs(PRs, PRs_prime))

    # 2. Filter Pass
    PRs = [pr for pr in PRs if pr[2] is not None]
    PRs_filtered = filter_PRs(PRs)
    PRs_filtered = fix_bibartie_issue_import_from(PRs_filtered)
    PR_filtered_no_assign = remove_assignments(PRs_filtered)

    # 3. Graph Assembly
    G, tuples = construct_bipartite_graph(PR_filtered_no_assign)

    return G, PR_filtered_no_assign, tuples
