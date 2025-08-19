"""
Create dataflow graphs from the notebooks in the JetBrains 10M dataset.

Source: https://blog.jetbrains.com/datalore/2020/12/17/we-downloaded-10-000-000-jupyter-notebooks-from-github-this-is-what-we-learned/

Utilize the file provided by the dataset, saved as data/ntbs_list.json.

Example run:

python ApexDAG/crawl_notebooks/mining/mine_jetbrains_dataset_dataflow_graphs.py --greedy --stop_index 110000 --start_index 0
"""

import os
import argparse
import time
import random
import traceback
import pandas as pd

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.crawl_notebooks.jetbrains_analysis.jetbrains_notebook_iterator import (
    JetbrainsNotebookIterator,
)

from dotenv import load_dotenv

load_dotenv()

BUCKET_URL = "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"
OUTPUT_DIR = "jetbrains_dfg_100k_new/"
JSON_FILE = "data/ntbs_list.json"
RESULTS_DIR = os.getenv("RESULTS_DIR")

FULL_OUTPUT_DIR = os.path.join(RESULTS_DIR, OUTPUT_DIR)


def mine_dataflows_on_jetbrains_dataset(args):
    stats = {}

    folder_dfg = os.path.join(FULL_OUTPUT_DIR, "execution_graphs")
    if not os.path.exists(folder_dfg):
        os.makedirs(folder_dfg)

    folder_code = os.path.join(FULL_OUTPUT_DIR, "code")
    if not os.path.exists(folder_code):
        os.makedirs(folder_code)

    jetbrains_iterator = JetbrainsNotebookIterator(
        JSON_FILE,
        BUCKET_URL,
        FULL_OUTPUT_DIR,
        log_file=f"notebook_processor_{args.start_index}_{args.stop_index}.log",
        start_index=args.start_index,
        stop_index=args.stop_index,
    )

    for filename, notebook_object in jetbrains_iterator:
        name = filename.replace(".ipynb", "")
        notebook_url = f"{BUCKET_URL}{filename}"
        stats[filename] = {
            "notebook_url": notebook_url,
            "loc": None,
            "dfg_extract_time": None,
            "execution_graph_time": None,
            "exception": None,
            "stacktrace": None,
        }

        try:
            execution_graph_start = time.time()
            notebook = Notebook(url=None, nb=notebook_object)
            notebook.create_execution_graph(greedy=args.greedy)
            execution_graph_end = time.time()
            stats[filename]["execution_graph_time"] = (
                execution_graph_end - execution_graph_start
            )
            stats[filename]["loc"] = notebook.loc()

            dfg_start_time = time.time()
            dfg = DataFlowGraph()
            dfg.parse_notebook(notebook)
            dfg.optimize()
            dfg_end_time = time.time()

            # Sampling logic based on the number of edges in the dataflow graph
            num_edges = len(dfg.get_edges())
            sample_this_notebook = False

            if num_edges < 50:
                # Small notebooks: 50% sampling rate
                if random.random() < 0.5:
                    sample_this_notebook = True
            elif 50 <= num_edges <= 250:
                # Mid-size notebooks: 100% sampling rate
                sample_this_notebook = True
            else: # num_edges > 250
                # Large notebooks: 75% sampling rate
                if random.random() < 0.75:
                    sample_this_notebook = True

            if sample_this_notebook:
                dfg.save_dfg(os.path.join(folder_dfg, f"{name}.execution_graph"))
                notebook.save_code(os.path.join(folder_code, f"{name}.code"))
            else:
                jetbrains_iterator.print(filename, f"Skipping notebook {filename} due to sampling.")

            if args.draw:
                dfg.draw(os.path.join("output", name, "dfg"))

            stats[filename]["dfg_extract_time"] = dfg_end_time - dfg_start_time

        except Exception as e:
            tb = traceback.format_exc()
            # if the code is not parseable because of syntax or identation error, 
            # i dont want to write the stacktrace to the log file, just say something like syntax error in X
            # I still want to write the stck trace to disk if I later wanna have a look but not to the log
            # TabError and UnicodeDecodeError are also caused by the notebook code
            if isinstance(e, (SyntaxError, IndentationError, TabError, UnicodeDecodeError)):
                jetbrains_iterator.print(filename, f"Syntax error in notebook {notebook_url} ({e.__class__.__name__})")
            else:
                jetbrains_iterator.print(filename, tb)  # this only prints to log!
            
            jetbrains_iterator.print(filename, f"Error in notebook {notebook_url}")
            stats[filename]["dfg_extract_time"] = -float("inf")

            folder = os.path.join(FULL_OUTPUT_DIR, "errors", name, "stacktraces")
            if not os.path.exists(folder):
                os.makedirs(folder)

            file_name = f"{name}.stacktrace"
            file_path = os.path.join(folder, file_name)

            with open(
                os.path.join(folder, "traceback.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(tb)

            stats[filename]["stacktrace"] = file_path
            stats[filename]["exception"] = e.__class__.__name__
            notebook.save_code(os.path.join(folder, f"{name}.code"))

    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    if not os.path.exists(FULL_OUTPUT_DIR):
        os.makedirs(FULL_OUTPUT_DIR, exist_ok=True)

    stats_df.to_csv(
        os.path.join(FULL_OUTPUT_DIR, "dfg_experiment_jetbrains.csv"), encoding="utf-8"
    )
    jetbrains_iterator.print(
        f"Succesfully extracted dataflow graphs for {stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0]}/{stats_df.shape[0]}"
    )
    if (
        stats_df[stats_df["dfg_extract_time"] > float("-inf")].shape[0]
        < stats_df.shape[0]
    ):
        jetbrains_iterator.print(
            f"Error types observed: {stats_df[stats_df['exception'].str.len() > 0]['exception'].unique()}"
        )
        jetbrains_iterator.print(
            f"Stacktraces for failed notebooks can be found in {os.path.join('output', 'stacktraces')}"
        )
    jetbrains_iterator.print(
        f"Mean execution graph creation time: {stats_df['execution_graph_time'].mean()}s"
    )
    jetbrains_iterator.print(
        f"Median execution graph creation time: {stats_df['execution_graph_time'].median()}s"
    )
    jetbrains_iterator.print(
        f"Mean DFG extraction time: {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['dfg_extract_time'].mean()}s"
    )
    jetbrains_iterator.print(
        f"Median DFG extraction time: {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['dfg_extract_time'].median()}s"
    )
    jetbrains_iterator.print(
        f"Mean LoC (statements): {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['loc'].mean()}"
    )
    jetbrains_iterator.print(
        f"Median LoC (statements): {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['loc'].median()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy algorithm to create execution graph",
    )
    parser.add_argument("--draw", action="store_true", help="Draw the data flow graph")
    parser.add_argument("--start_index", type=int, default=0, help="Start index")
    parser.add_argument("--stop_index", type=int, default=110000, help="End index")
    args = parser.parse_args()

    mine_dataflows_on_jetbrains_dataset(args)
