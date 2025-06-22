import logging

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_ast_graph import PythonASTGraph as ASTGraph


def ast_graph_test(args, logger: logging.Logger) -> None:
    logger.info("Using notebook %s", args.notebook)

    notebook = Notebook(args.notebook, cell_window_size=args.window)
    notebook.create_execution_graph(greedy=args.greedy)

    # static code analysis
    ast_graph = ASTGraph()
    ast_graph.parse_notebook(notebook)
    
    logger.info("Number of nodes: %s", str(len(ast_graph.get_nodes())))
    logger.info("Number of edges: %s", str(len(ast_graph.get_edges())))
    max_depth = 5
    logger.info("Leaf to leaf paths (max_depth=%s): %s", str(max_depth), str(len(ast_graph.get_t2t_paths(max_depth=max_depth))))
    logger.info("Leaf nodes: %s", str(len(ast_graph.get_leaf_nodes())))
    
    ast_graph.draw()