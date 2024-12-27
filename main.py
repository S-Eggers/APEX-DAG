import sys
import time
import matplotlib

from ApexDAG.argparser import argparser

from ApexDAG.experiments.analyze_dataset import analyze_kaggle_dataset
from ApexDAG.experiments.dataflow_graph import data_flow_graph_test
from ApexDAG.experiments.ast_graph_test import ast_graph_test
from ApexDAG.experiments.watch import watch
from ApexDAG.experiments.mine_dataflows import mine_dataflows_on_kaggle_dataset
from ApexDAG.util.logging import setup_logging


def main(argv=None):
    logger = setup_logging("main", True)
    args = argparser.parse_args(argv)
    start_time = time.time()
    experiments = {
        "ast_graph_test": ast_graph_test,
        "analyze_kaggle_dataset": analyze_kaggle_dataset,
        "data_flow_graph_test": data_flow_graph_test,
        "mine_dataflows_on_kaggle_dataset": mine_dataflows_on_kaggle_dataset,
        "watch": watch,
    }
    if args.experiment not in experiments:
        raise ValueError(f"Unknown experiment {args.experiment}")

    experiments[args.experiment](args, logger)
    end_time = time.time()
    logger.debug(f"Pipeline took {end_time - start_time}s")


if __name__ == "__main__":
    matplotlib.use("agg")
    main(sys.argv[1:])
