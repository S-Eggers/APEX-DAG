import os
import re
import time
import traceback
import pandas as pd

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.scripts.crawl_notebooks.jetbrains_analysis.jetbrains_notebook_iterator import JetbrainsNotebookIterator


BUCKET_URL= "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"
OUTPUT_DIR = 'output_dataflow_jetbrains'


def mine_dataflows_on_kaggle_dataset(args):
    jetbrains_iterator = JetbrainsNotebookIterator(
        JSON_FILE = "data/ntbs_list.json",
        BUCKET_URL= BUCKET_URL, 
        SAVE_DIR= "data/notebooks", 
        log_file = 'log_mine_jetbrains_notebooks.log',
        start_index = 0,
        stop_index = 10
        
    )
    stats = {"notebook_url": [], "loc": [], "dfg_extract_time": [], "execution_graph_time": [], "exception": [], "stacktrace": []}
    for filename, notebook_object in jetbrains_iterator:
        name = filename.replace(".ipynb", "")
        notebook_url = f"{BUCKET_URL}{filename}"
        stats["notebook_url"].append(notebook_url)
        
        try:
            execution_graph_start = time.time()
            notebook = Notebook(url = None, nb = notebook_object)
            notebook.create_execution_graph(greedy=args.greedy)
            execution_graph_end = time.time()
            stats["execution_graph_time"].append(execution_graph_end - execution_graph_start)
            stats["loc"].append(notebook.loc())

            dfg_start_time = time.time()
            dfg = DataFlowGraph()
            dfg.parse_notebook(notebook)
            dfg.optimize()
            dfg_end_time = time.time()
                
            if args.draw:
                dfg.draw(os.path.join("output", name, "dfg"))
                
            stats["dfg_extract_time"].append(dfg_end_time - dfg_start_time)
            stats["stacktrace"].append(None)
            stats["exception"].append(None)

        except Exception as e:
            notebook_path = f"{notebook_path}"
            tb = traceback.format_exc()
            jetbrains_iterator.print(tb) # this only prints to log!
            jetbrains_iterator.print(f"Error in notebook {notebook_path}")
            stats["dfg_extract_time"].append(-float("inf"))
                
            folder = os.path.join("output", name, "stacktraces")
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            file_name = f"{name}.stacktrace"
            file_path = os.path.join(folder, file_name)
            with open(file_path, "w+") as f:
                f.write(str(tb))
            stats["stacktrace"].append(file_path)
            stats["exception"].append(e.__class__.__name__)
            notebook.save_code(os.path.join(folder, f"{name}.code"))

    stats_df = pd.DataFrame(stats)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "dfg_experiment_jetbrains.csv"))
    jetbrains_iterator.print(f"Succesfully extracted dataflow graphs for {stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0]}/{stats_df.shape[0]}")
    if stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0] < stats_df.shape[0]:
        jetbrains_iterator.print(f"Error types observed: {stats_df[stats_df['exception'].str.len()>0]['exception'].unique()}")
        jetbrains_iterator.print(f"Stacktraces for failed notebooks can be found in {os.path.join('output', 'stacktraces')}")
    jetbrains_iterator.print(f"Mean execution graph creation time: {stats_df['execution_graph_time'].mean()}s")
    jetbrains_iterator.print(f"Median execution graph creation time: {stats_df['execution_graph_time'].median()}s")
    jetbrains_iterator.print(f"Mean DFG extraction time: {stats_df[stats_df['dfg_extract_time']>float('-inf')]['dfg_extract_time'].mean()}s")
    jetbrains_iterator.print(f"Median DFG extraction time: {stats_df[stats_df['dfg_extract_time']>float('-inf')]['dfg_extract_time'].median()}s")
    jetbrains_iterator.print(f"Mean LoC (statements): {stats_df[stats_df['dfg_extract_time']>float('-inf')]['loc'].mean()}")
    jetbrains_iterator.print(f"Median LoC (statements): {stats_df[stats_df['dfg_extract_time']>float('-inf')]['loc'].median()}")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", action="store_true", help="Use greedy algorithm to create execution graph")
    parser.add_argument("--draw", action="store_true", help="Draw the data flow graph")
    args = parser.parse_args()
    
    #todo remove
    args.greedy = True
    
    mine_dataflows_on_kaggle_dataset(args)