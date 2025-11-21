import os
import time
import logging

from ApexDAG.notebook import Notebook
from ApexDAG.sca.constants import NODE_TYPES, EDGE_TYPES
from ApexDAG.sca.graph_utils import debug_graph
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph


def data_flow_graph_test(args, logger: logging.Logger) -> None:
    Notebook.VERBOSE = True
    DataFlowGraph.VERBOSE = False
    logger.debug(f"Using notebook {args.notebook}")

    start_time = time.time()
    notebook = Notebook(args.notebook, cell_window_size=args.window)
    notebook.create_execution_graph(greedy=args.greedy)
    end_time = time.time()
    notebook.print_code()
    logger.debug(f"Building notebook execution graph took {end_time - start_time}s")

    start_time = time.time()
    dfg = DataFlowGraph()
    dfg.parse_notebook(notebook)
    end_time = time.time()
    logger.debug(f"Building dataflow graph took {end_time - start_time}s")

    start_time = time.time()
    dfg.draw(save_path = os.path.join("output", "dfg_images_unoptimized.png"))
    end_time = time.time()
    logger.debug(f"Drawing dataflow graph images took {end_time - start_time}s")


    start_time = time.time()
    dfg.optimize()
    end_time = time.time()
    logger.debug(f"Optimizing dataflow graph took {end_time - start_time}s")

    start_time = time.time()
    dfg.draw(save_path = os.path.join("output", "dfg_images.png"))
    end_time = time.time()
    logger.debug(f"Drawing dataflow graph images took {end_time - start_time}s")

    G = dfg.get_graph()
    start_time = time.time()
    in_out_file_path = os.path.join("output", "dfg.gml")
    debug_graph(
        G,
        in_out_file_path,
        in_out_file_path,
        NODE_TYPES,
        EDGE_TYPES,
        save_prev=args.save_prev,
        verbose=True,
    )
    end_time = time.time()

    dfg.save_dfg(os.path.join(os.getcwd(), "output", "dfg.pkl"))
    logger.debug(f"Debugging dataflow graph took {end_time - start_time}s")
