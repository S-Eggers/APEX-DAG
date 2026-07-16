from __future__ import annotations

import argparse
from pathlib import Path

from SystemX.pipeline._shared import VAMSA_KB_PATH
from SystemX.util.logger import configure_systemx_logger

from .experiment import render_saved_results, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Vamsa static-ablation cross-dataset generalization experiment")
    parser.add_argument("--kb_csv", type=Path, default=VAMSA_KB_PATH, help="Canonical Vamsa knowledge-base CSV.")
    parser.add_argument("--output_path", type=Path, default=Path("output/results/vamsa_generalization.json"))
    parser.add_argument("--kb_output_dir", type=Path, default=Path("output/vamsa_generalization/kb"))
    parser.add_argument("--figure_dir", type=Path, default=Path("output/figures"))
    parser.add_argument("--plot_only", action="store_true", help="Regenerate figures from --output_path without rerunning evaluation.")
    args = parser.parse_args()

    configure_systemx_logger()
    if args.plot_only:
        render_saved_results(args.output_path, args.figure_dir)
    else:
        run_experiment(args.kb_csv, args.output_path, args.kb_output_dir, args.figure_dir)


if __name__ == "__main__":
    main()
