import os
import json
import argparse
import time
import traceback
import pandas as pd

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.scripts.crawl_notebooks.github_crawler.paginated_notebook_iterator import PaginatedNotebookIterator


SAVE_DIR = "data/notebooks/github"
GITHUB_API_URL = "https://api.github.com/search/code"
OUTPUT_DIR = 'output/output_dataflow_github'
JSON_FILE = "data/ntbs_list.json"
SAVE_DIR = "output_dataflow_github/"

def mine_dataflows_on_github_dataset(args):
    stats = {}

    github_iterator = PaginatedNotebookIterator(
        query="extension:ipynb",
        per_page=100,
        max_results=args.stop_index - args.start_index + 1,
        search_url=GITHUB_API_URL,
        log_file=f'notebook_graph_miner.log'
    )

    # save folder for exe graphs
    folder_dfg = os.path.join(OUTPUT_DIR, "execution_graphs")
    if not os.path.exists(folder_dfg):
        os.makedirs(folder_dfg)
        
    folder_datafiles = os.path.join(OUTPUT_DIR, "datafiles")
    if not os.path.exists(folder_datafiles):
        os.makedirs(folder_datafiles)
        
    datafiles = {}
        
    for filename, notebook_object in github_iterator:
        
        if notebook_object.get('notebook_content', None) is not None:
            if notebook_object.get('data_files', None) is not None:
                datafiles_current = notebook_object['data_files']
                datafiles[filename] = datafiles_current
            notebook_object = notebook_object['notebook_content']
            
        name = filename.replace(".ipynb", "").replace("https://github.com/", "").replace("/", "_").replace(".", "_")
        notebook_url = filename
        stats[filename] = {
            "notebook_url": notebook_url,
            'name': name, # names are a bit more complicated in this example, thus including them, seems better than not doing so
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
            
            # save execution graph to disk
            dfg.save_dfg(os.path.join(folder_dfg, f"{name}.execution_graph"))

            if args.draw:
                dfg.draw(os.path.join("output", name, "dfg"))

            stats[filename]["dfg_extract_time"] = dfg_end_time - dfg_start_time

        except Exception as e:
            tb = traceback.format_exc()
            github_iterator.print(filename, tb)  # this only prints to log!
            github_iterator.print(filename, f"Error in notebook {notebook_url}")
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
            notebook.save_code(os.path.join(folder, f"{name[:10]}.code"))

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # save datafiles to disk
    with open(os.path.join(folder_datafiles, "datafiles.json"), 'w') as f:
        json.dump(datafiles, f, indent=4)
        
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "dfg_experiment_github.csv"), encoding="utf-8")
    github_iterator.print(f"Succesfully extracted dataflow graphs for {stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0]}/{stats_df.shape[0]}")
    if stats_df[stats_df['dfg_extract_time'] > float('-inf')].shape[0] < stats_df.shape[0]:
        github_iterator.print(f"Error types observed: {stats_df[stats_df['exception'].str.len() > 0]['exception'].unique()}")
        github_iterator.print(f"Stacktraces for failed notebooks can be found in {os.path.join('output', 'stacktraces')}")
    github_iterator.print(f"Mean execution graph creation time: {stats_df['execution_graph_time'].mean()}s")
    github_iterator.print(f"Median execution graph creation time: {stats_df['execution_graph_time'].median()}s")
    github_iterator.print(f"Mean DFG extraction time: {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['dfg_extract_time'].mean()}s")
    github_iterator.print(f"Median DFG extraction time: {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['dfg_extract_time'].median()}s")
    github_iterator.print(f"Mean LoC (statements): {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['loc'].mean()}")
    github_iterator.print(f"Median LoC (statements): {stats_df[stats_df['dfg_extract_time'] > float('-inf')]['loc'].median()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", action="store_true", help="Use greedy algorithm to create execution graph")
    parser.add_argument("--draw", action="store_true", help="Draw the data flow graph")
    parser.add_argument("--start_index", default=0, help="Start index")
    parser.add_argument("--stop_index", default=100, help="End index")
    args = parser.parse_args()

    mine_dataflows_on_github_dataset(args)