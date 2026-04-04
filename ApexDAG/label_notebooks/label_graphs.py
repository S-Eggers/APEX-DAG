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

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define log format
log_format = "%(asctime)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

# Create log directory if it doesn't exist
log_dir = "jetbrains_dfg_100k_new/logs"
os.makedirs(log_dir, exist_ok=True)

# Create file handler
log_file = os.path.join(log_dir, "graph_labeling.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.propagate = False


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
        default="jetbrains_dfg_100k_new/execution_graphs",
        help="Path to the input graph files",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="jetbrains_dfg_100k_new/labelled_execution_graphs",
        help="Path to save labeled graph files",
    )

    args = parser.parse_args()

    config = load_config(args.config_path)

    source_path = os.path.join(os.getcwd(), args.source_path)
    if not os.path.exists(source_path):
        logger.error(f"Source path '{source_path}' does not exist.")
        exit(1)

    target_dir = os.path.join(os.getcwd(), args.target_path)
    os.makedirs(target_dir, exist_ok=True)

    logger.info(f"Initializing processed files set from {target_dir}...")
    processed_files = set(os.listdir(target_dir))
    logger.info(f"Found {len(processed_files)} already processed files.")

    try:
        while True:
            files = [f for f in os.listdir(source_path) if f.endswith(".execution_graph")]
            
            new_files = [f for f in files if f not in processed_files]

            if not new_files:
                logger.info("No new files found. Waiting for new files...")
                time.sleep(60)
                continue

            for filename in tqdm(new_files, desc="Processing graph files"):
                output_directory = os.path.join(os.getcwd(), args.target_path)
                output_file = os.path.join(output_directory, filename)

                if os.path.exists(output_file):
                    logger.info(f"Skipping already processed file: {filename}")
                    processed_files.add(filename)
                    continue

                graph_file_path = os.path.join(source_path, filename)
                logger.info(f"Processing file: {graph_file_path}")

                try:
                    G = load_graph(graph_file_path)
                    code_file_path = get_code_file_path(graph_file_path)

                    labeler = GraphLabeler(config, graph_file_path, code_file_path, logger)
                    G, G_with_context = labeler.label_graph()
                    total_tokens_used = labeler.get_total_tokens_used()
                    logger.info(f"Total tokens used: {total_tokens_used}")
                    config.max_tokens = config.max_tokens - total_tokens_used
                    logger.info(f"Remaining tokens: {config.max_tokens}")
                    if config.max_tokens < 0:
                        raise KeyboardInterrupt()

                    os.makedirs(output_directory, exist_ok=True)
                    nx.write_gml(G, output_file)

                    logger.info(f"Saved labeled graph to: {output_file}")
                    processed_files.add(filename)
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}", exc_info=True)
            
            logger.info("Finished processing all new files. Waiting for new files...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Graph labeling process interrupted by user.")
