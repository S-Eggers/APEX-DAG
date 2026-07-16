from __future__ import annotations

import sys

from . import evaluate as _evaluate
from . import plot_prevalence as _plot
from . import run_analysis as _analyze
from . import seed_benchmark as _seed

_SUBCOMMANDS = {
    "seed": _seed.main,
    "evaluate": _evaluate.main,
    "analyze": _analyze.main,
    "plot": _plot.main,
}

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _SUBCOMMANDS:
        sys.stderr.write(f"usage: python -m SystemX.experiment.leakage {{{'|'.join(_SUBCOMMANDS)}}} ...\n")
        return 2
    return _SUBCOMMANDS[argv[0]](argv[1:])

if __name__ == "__main__":
    raise SystemExit(main())
