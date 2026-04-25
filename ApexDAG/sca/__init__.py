"""
This module initializes the SCA (Static Code Analysis) package of the ApexDAG library.
"""
# Corse parser
# Legacy/Public AST Utilities
from ApexDAG.sca.ast_utils import flatten_list, get_operator_description

# Constants
from ApexDAG.sca.constants import (
    AST_EDGE_TYPES,
    AST_NODE_TYPES,
    DOMAIN_EDGE_TYPES,
    DOMAIN_NODE_TYPES,
    EDGE_TYPES,
    NODE_TYPES,
    REVERSE_AST_EDGE_TYPES,
    REVERSE_AST_NODE_TYPES,
    REVERSE_DOMAIN_EDGE_TYPES,
    REVERSE_DOMAIN_NODE_TYPES,
    REVERSE_EDGE_TYPES,
    REVERSE_NODE_TYPES,
    VERBOSE,
)

# Graph Utilities
from ApexDAG.sca.graph_utils import (
    convert_multidigraph_to_digraph,
    debug_graph,
    get_all_subgraphs,
    get_subgraph,
    load_graph,
    save_graph,
)

# Models
from ApexDAG.sca.models import GraphEdge, GraphNode
from ApexDAG.sca.py_ast_graph import PythonASTGraph
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph

__all__ = [
    # Core Parsers
    "PythonASTGraph",
    "PythonDataFlowGraph",

    # Data Contracts
    "GraphNode",
    "GraphEdge",

    # Taxonomy & Constants
    "VERBOSE",
    "NODE_TYPES",
    "EDGE_TYPES",
    "DOMAIN_EDGE_TYPES",
    "DOMAIN_NODE_TYPES",
    "AST_NODE_TYPES",
    "AST_EDGE_TYPES",
    "REVERSE_AST_NODE_TYPES",
    "REVERSE_AST_EDGE_TYPES",
    "REVERSE_NODE_TYPES",
    "REVERSE_EDGE_TYPES",
    "REVERSE_DOMAIN_NODE_TYPES",
    "REVERSE_DOMAIN_EDGE_TYPES",

    # Utilities
    "convert_multidigraph_to_digraph",
    "get_subgraph",
    "get_all_subgraphs",
    "debug_graph",
    "save_graph",
    "load_graph",
    "get_operator_description",
    "flatten_list",
]
