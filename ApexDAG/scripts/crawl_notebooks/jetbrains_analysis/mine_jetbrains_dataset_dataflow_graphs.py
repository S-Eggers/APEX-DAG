import os
import re
import argparse
import time
import traceback
import pandas as pd

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.scripts.crawl_notebooks.jetbrains_analysis.jetbrains_notebook_iterator import JetbrainsNotebookIterator


BUCKET_URL= "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"
OUTPUT_DIR = 'output/output_dataflow_jetbrains'
JSON_FILE = "data/ntbs_list.json"
SAVE_DIR = "output_dataflow_jetbrains/"

def mine_dataflows_on_jetbrains_dataset(args):
    stats = {}

    jetbrains_iterator = JetbrainsNotebookIterator(
        JSON_FILE, BUCKET_URL, SAVE_DIR, log_file=f'notebook_processor_{args.start_index}_{args.stop_index}.log',
        start_index=args.start_index, stop_index=args.stop_index
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
            "stacktrace": None
        }

        try:
            execution_graph_start = time.time()
            notebook = Notebook(url=None, nb=notebook_object)
            notebook.create_execution_graph(greedy=args.greedy)
            execution_graph_end = time.time()
            stats[filename]["execution_graph_time"] = execution_graph_end - execution_graph_start
            stats[filename]["loc"] = notebook.loc()

            dfg_start_time = time.time()
            dfg = DataFlowGraph()
            dfg.parse_notebook(notebook)
            dfg.optimize()
            dfg_end_time = time.time()

            if args.draw:
                dfg.draw(os.path.join("output", name, "dfg"))

            stats[filename]["dfg_extract_time"] = dfg_end_time - dfg_start_time

        except Exception as e:
            tb = traceback.format_exc()
            jetbrains_iterator.print(filename, tb)  # this only prints to log!
            jetbrains_iterator.print(filename, f"Error in notebook {notebook_url}")
            stats[filename]["dfg_extract_time"] = -float("inf")

            folder = os.path.join(OUTPUT_DIR, "errors", name, "stacktraces")
            if not os.path.exists(folder):
                os.makedirs(folder)

            file_name = f"{name}.stacktrace"
            file_path = os.path.join(folder, file_name)

            with open(os.path.join(folder, "traceback.txt"), "w", encoding="utf-8") as f:
                f.write(tb)

            stats[filename]["stacktrace"] = file_path
            stats[filename]["exception"] = e.__class__.__name__
            notebook.save_code(os.path.join(folder, f"{name}.code"))

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats_df.to_csv(os.path.join(OUTPUT_DIR, "dfg_experiment_jetbrains.csv"), encoding="utf-8")
    jetbrains_iterator.print(f"Succesfully extracted dataflow graphs for {stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0]}/{stats_df.shape[0]}")
    if stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0] < stats_df.shape[0]:
        jetbrains_iterator.print(f"Error types observed: {stats_df[stats_df['exception'].str.len() > 0]['exception'].unique()}")
        jetbrains_iterator.print(f"Stacktraces for failed notebooks can be found in {os.path.join('output', 'stacktraces')}")
    jetbrains_iterator.print(f"Mean execution graph creation time: {stats_df['execution_graph_time'].mean()}s")
    jetbrains_iterator.print(f"Median execution graph creation time: {stats_df['execution_graph_time'].median()}s")
    jetbrains_iterator.print(f"Mean DFG extraction time: {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['dfg_extract_time'].mean()}s")
    jetbrains_iterator.print(f"Median DFG extraction time: {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['dfg_extract_time'].median()}s")
    jetbrains_iterator.print(f"Mean LoC (statements): {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['loc'].mean()}")
    jetbrains_iterator.print(f"Median LoC (statements): {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['loc'].median()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", action="store_true", help="Use greedy algorithm to create execution graph")
    parser.add_argument("--draw", action="store_true", help="Draw the data flow graph")
    parser.add_argument("--start_index", default=0, help="Start index")
    parser.add_argument("--stop_index", default=1000, help="End index")
    args = parser.parse_args()

    mine_dataflows_on_jetbrains_dataset(args)