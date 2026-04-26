import logging
import sys


def configure_apexdag_logger(
    jupyter_logger: logging.Logger | None = None,
) -> logging.Logger:
    base_logger = logging.getLogger("ApexDAG")
    base_logger.setLevel(logging.INFO)

    if base_logger.hasHandlers():
        base_logger.handlers.clear()

    if jupyter_logger:
        for handler in jupyter_logger.handlers:
            base_logger.addHandler(handler)

        base_logger.propagate = False
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        base_logger.addHandler(handler)
        base_logger.propagate = False

    return base_logger
