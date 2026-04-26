import sys
import time

import matplotlib
from ApexDAG.argparser import argparser
from ApexDAG.util.logging import setup_logging


def main(argv=None) -> None:
    logger = setup_logging("main", True)
    args = argparser.parse_args(argv)

    start_time = time.time()
    match args.experiment:
        case "evaluate":
            print("WiP")
        case _:
            raise ValueError(f"Unknown experiment {args.experiment}")

    end_time = time.time()
    logger.debug(f"Pipeline took {end_time - start_time}s")


if __name__ == "__main__":
    import fasttext.util

    fasttext.util.download_model("en", if_exists="ignore")
    matplotlib.use("agg")

    main(sys.argv[1:])
