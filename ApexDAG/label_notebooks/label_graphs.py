import argparse
import os
import networkx as nx
import logging
import time
from tqdm import tqdm
from ApexDAG.label_notebooks.label_graph import GraphLabeler
from ApexDAG.label_notebooks.utils import load_config
from ApexDAG.sca.graph_utils import load_graph
from dotenv import load_dotenv

load_dotenv()

log_format = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[logging.FileHandler("graph_labeling.log"), logging.StreamHandler()],
)


def get_code_file_path(filename):
    """
    Get code file for a given execution graph file.
    """
    code_filename = filename.replace(".execution_graph", ".code")
    code_filename = code_filename.replace("execution_graphs", "code")
    return code_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        default="ApexDAG/label_notebooks/config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/home/eggers/data/apexdag_results/jetbrains_dfg_100k_new/execution_graphs",
        help="Path to the input graph files",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="/home/eggers/data/apexdag_results/jetbrains_dfg_100k_new_labeled/execution_graphs",
        help="Path to save labeled graph files",
    )

    args = parser.parse_args()

    config = load_config(args.config_path)
    config.max_tokens = 1000000000

    if not os.path.exists(args.source_path):
        logging.error(f"Source path '{args.source_path}' does not exist.")
        exit(1)

    os.makedirs(args.target_path, exist_ok=True)
    files = [f for f in os.listdir(args.source_path) if f.endswith(".execution_graph")]

    for filename in tqdm(files, desc="Processing graph files"):
        if filename.endswith(".execution_graph"):
            graph_file_path = os.path.join(args.source_path, filename)
            logging.info(f"Processing file: {graph_file_path}")

            try:
                G = load_graph(graph_file_path)
                code_file_path = get_code_file_path(graph_file_path)

                labeler = GraphLabeler(config, graph_file_path, code_file_path)
                G, G_with_context = labeler.label_graph()

                output_directory = args.target_path
                os.makedirs(output_directory, exist_ok=True)

                output_file = os.path.join(output_directory, filename)
                nx.write_gml(G, output_file)

                logging.info(f"Saved labeled graph to: {output_file}")
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}", exc_info=True)
