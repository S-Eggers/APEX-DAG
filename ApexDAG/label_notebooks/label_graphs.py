import argparse
import os
import networkx as nx
import logging
import time
from tqdm import tqdm
from ApexDAG.label_notebooks.label_graph import label_graph
from ApexDAG.label_notebooks.utils import load_config
from ApexDAG.sca.graph_utils import load_graph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging to both file and stdout
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("graph_labeling.log"),  # Log to a file
        logging.StreamHandler()  # Log to stdout
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path", type=str, default="ApexDAG/label_notebooks/config.yaml", help="Path to the configuration file")
    parser.add_argument("--source_path", type=str, default="/home/eggers/data/apexdag_results/github_dfg_100k/execution_graphs", help="Path to the input graph files")
    parser.add_argument("--target_path", type=str, default="/home/eggers/data/apexdag_results/github_dfg_100k_labelled/execution_graphs", help="Path to save labeled graph files")
    
    args = parser.parse_args()
    
    config = load_config(args.config_path)
    
    if not os.path.exists(args.source_path):
        logging.error(f"Source path '{args.source_path}' does not exist.")
        exit(1)

    os.makedirs(args.target_path, exist_ok=True)
    files = [f for f in os.listdir(args.source_path) if f.endswith(".execution_graph")]
    
    for filename in tqdm(files, desc="Processing graph files"):
        if filename.endswith(".execution_graph"):
            file_path = os.path.join(args.source_path, filename)
            logging.info(f"Processing file: {file_path}")
            
            try:
                G = load_graph(file_path)
                G, G_with_context = label_graph(config, G)
                    
                output_directory = args.target_path
                os.makedirs(output_directory, exist_ok=True)
                    
                output_file = os.path.join(output_directory, filename)
                nx.write_gml(G, output_file)
                    
                logging.info(f"Saved labeled graph to: {output_file}")
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
                time.sleep(10)