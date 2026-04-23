"""
This module initializes the SCA (Static Code Analysis) package of the ApexDAG library.
"""
# Corse parser
from ApexDAG.sca.py_ast_graph import PythonASTGraph
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph

# Models
from ApexDAG.sca.models import GraphNode, GraphEdge

# Constants
from ApexDAG.sca.constants import (
    VERBOSE,
    NODE_TYPES,
    EDGE_TYPES,
    REVERSE_NODE_TYPES,
    REVERSE_EDGE_TYPES,
    DOMAIN_EDGE_TYPES,
    DOMAIN_NODE_TYPES,
    REVERSE_DOMAIN_EDGE_TYPES,
    AST_NODE_TYPES,
    AST_EDGE_TYPES
)

# Graph Utilities
from ApexDAG.sca.graph_utils import (
    convert_multidigraph_to_digraph,
    get_subgraph,
    get_all_subgraphs,
    debug_graph,
    save_graph,
    load_graph,
)

# Legacy/Public AST Utilities
from ApexDAG.sca.ast_utils import get_operator_description, flatten_list

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
    "REVERSE_NODE_TYPES",
    "REVERSE_EDGE_TYPES",
    "DOMAIN_EDGE_TYPES",
    "DOMAIN_NODE_TYPES",
    "REVERSE_DOMAIN_EDGE_TYPES",
    "AST_NODE_TYPES",
    "AST_EDGE_TYPES",
    
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