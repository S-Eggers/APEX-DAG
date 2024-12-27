import os
import re
import time
import traceback
import pandas as pd

from ApexDAG.scripts.notebook import Notebook
from ApexDAG.scripts.ast.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.scripts.util.kaggle_dataset_iterator import KaggleDatasetIterator


def mine_dataflows_on_kaggle_dataset(args, logger):
    kaggle_iterator = KaggleDatasetIterator(os.path.join(os.getcwd(), "data", "raw", "notebooks"))
    stats = {"notebook": [], "loc": [], "competition": [], "dfg_extract_time": [], "execution_graph_time": [], "exception": [], "stacktrace": []}
    for competition in kaggle_iterator:
        for notebook_file in competition["ipynb_files"]:
            name = re.sub(r'\W+', '', f"{competition['json_file']['name']}_{notebook_file}")
            stats["competition"].append(competition["json_file"]["url"])
            stats["notebook"].append(notebook_file)
            try:
                execution_graph_start = time.time()
                notebook_path = os.path.join(competition["subfolder_path"], notebook_file)
                notebook = Notebook(notebook_path)
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
                kaggle_iterator.print(tb)
                kaggle_iterator.print(f"Error in notebook {notebook_path}")
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
    if not os.path.exists("output"):
        os.makedirs("output", exist_ok=True)
    stats_df.to_csv(os.path.join("output", "dfg_experiment.csv"))
    kaggle_iterator.print(f"Succesfully extracted dataflow graphs for {stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0]}/{stats_df.shape[0]}")
    if stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0] < stats_df.shape[0]:
        kaggle_iterator.print(f"Error types observed: {stats_df[stats_df['exception'].str.len()>0]['exception'].unique()}")
        kaggle_iterator.print(f"Stacktraces for failed notebooks can be found in {os.path.join('output', 'stacktraces')}")
    kaggle_iterator.print(f"Mean execution graph creation time: {stats_df['execution_graph_time'].mean()}s")
    kaggle_iterator.print(f"Median execution graph creation time: {stats_df['execution_graph_time'].median()}s")
    kaggle_iterator.print(f"Mean DFG extraction time: {stats_df[stats_df['dfg_extract_time']>float('-inf')]['dfg_extract_time'].mean()}s")
    kaggle_iterator.print(f"Median DFG extraction time: {stats_df[stats_df['dfg_extract_time']>float('-inf')]['dfg_extract_time'].median()}s")
    kaggle_iterator.print(f"Mean LoC (statements): {stats_df[stats_df['dfg_extract_time']>float('-inf')]['loc'].mean()}")
    kaggle_iterator.print(f"Median LoC (statements): {stats_df[stats_df['dfg_extract_time']>float('-inf')]['loc'].median()}")
