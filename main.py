import sys
import time
import matplotlib

from ApexDAG.argparser import argparser
from ApexDAG.util.logging import setup_logging


def main(argv=None):
    logger = setup_logging("main", True)
    args = argparser.parse_args(argv)

    start_time = time.time()
    match args.experiment:
        case "ast_graph_test":
            from ApexDAG.experiments.ast_graph_test import ast_graph_test

            ast_graph_test(args, logger)
        case "analyze_kaggle_dataset":
            from ApexDAG.experiments.analyze_dataset import analyze_kaggle_dataset

            analyze_kaggle_dataset(args, logger)
        case "data_flow_graph_test":
            from ApexDAG.experiments.dataflow_graph import data_flow_graph_test

            data_flow_graph_test(args, logger)
        case "mine_dataflows_on_kaggle_dataset":
            from ApexDAG.experiments.mine_dataflows import (
                mine_dataflows_on_kaggle_dataset,
            )

            mine_dataflows_on_kaggle_dataset(args, logger)
        case "watch":
            from ApexDAG.experiments.watch import watch

            watch(args, logger)
        case "pretrain":
            from ApexDAG.experiments.pretrain import pretrain_gat

            pretrain_gat(args, logger)
        case "finetune":
            from ApexDAG.experiments.finetune import finetune_gat

            finetune_gat(args, logger)
        case _:
            raise ValueError(f"Unknown experiment {args.experiment}")

    end_time = time.time()
    logger.debug(f"Pipeline took {end_time - start_time}s")


if __name__ == "__main__":
    import fasttext.util

    fasttext.util.download_model("en", if_exists="ignore")
    matplotlib.use("agg")

    main(sys.argv[1:])
