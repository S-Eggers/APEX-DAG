import os

from ApexDAG.scripts.notebook import Notebook
from ApexDAG.scripts.ast.py_ast_graph import PythonASTGraph as ASTGraph


def ast_graph_test(args, logger):
    print(f"Using notebook {args.notebook}")

    notebook_path = os.path.join(os.getcwd(), "data", "raw", args.notebook)
    notebook = Notebook(notebook_path, cell_window_size=args.window)
    notebook.create_execution_graph(greedy=args.greedy)

    # static code analysis
    ast_graph = ASTGraph()
    ast_graph.parse_notebook(notebook)
    
    print("Number of nodes:", len(ast_graph.get_nodes()))
    print("Number of edges:", len(ast_graph.get_edges()))
    max_depth = 5
    print(f"Leaf to leaf paths (max_depth={max_depth}):", len(ast_graph.get_t2t_paths(max_depth=max_depth)))
    print("Leaf nodes:", len(ast_graph.get_leaf_nodes()))
    
    ast_graph.draw()