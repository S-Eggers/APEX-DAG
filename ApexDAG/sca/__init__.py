"""
This module initializes the SCA (Static Code Analysis) package of the ApexDAG library.
"""
from ApexDAG.sca.ast_graph import ASTGraph
from ApexDAG.sca.constants import (
    VERBOSE,
    NODE_TYPES,
    EDGE_TYPES,
    REVERSE_NODE_TYPES,
    REVERSE_EDGE_TYPES
)
from ApexDAG.sca.graph_utils import (
    convert_multidigraph_to_digraph,
    get_subgraph,
    get_all_subgraphs,
    debug_graph,
    save_graph,
    load_graph
)
from ApexDAG.sca.py_util import get_operator_description, flatten_list

__all__ = [
    "ASTGraph",
    "VERBOSE",
    "NODE_TYPES",
    "EDGE_TYPES",
    "REVERSE_NODE_TYPES",
    "REVERSE_EDGE_TYPES",
    "convert_multidigraph_to_digraph",
    "get_subgraph",
    "get_all_subgraphs",
    "debug_graph",
    "save_graph",
    "load_graph",
    "get_operator_description",
    "flatten_list"
]
