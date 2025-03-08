from argparse import ArgumentParser

argparser = ArgumentParser()

argparser.add_argument("-n", "--notebook", type=str, default="./data/raw/demo.ipynb", help="Jupyter notebook filename")
argparser.add_argument("-w", "--window", type=int, default=1, help="Cell window size for iterating the notebook cells")
argparser.add_argument("-g", "--greedy", action="store_true", help="Greedy notebook cell parsing")
argparser.add_argument("-s", "--save_prev", action="store_true", help="Store previous graph for debugging")
argparser.add_argument("-e", "--experiment", type=str, default="data_flow_graph_test", help="Experiment to run")
argparser.add_argument("-d", "--draw", action="store_true", help="Draw AST-/Dataflow Graphs with graph_viz and matplotlib")
argparser.add_argument("-c", "--checkpoint_path", type=str, default=None, help="Path to checkpoint directory")
argparser.add_argument("-cp", "--config_path", type=str, default="ApexDAG/experiments/configs/pretrain/default.yaml", help="Config file.")