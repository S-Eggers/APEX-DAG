import os
import json
import argparse
import time
import traceback
import pandas as pd

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph
from ApexDAG.crawl_notebooks.github_crawler.github_repository_notebook_iterator import GithubRepositoryNotebookIterator
from ApexDAG.crawl_notebooks.github_crawler.github_repository_crawler import GitHubRepositoryCrawler

GITHUB_API_URL = "https://api.github.com/search/code"
OUTPUT_DIR = '/home/eggers/data/apexdag_results/github_dfg_100k_new/'


def mine_dataflows_on_github_dataset(args):
    stats = {}

    github_iterator = GithubRepositoryNotebookIterator(
        max_results=args.stop_index, # get 100K at max this is not to be expected
        notebook_paths=args.notebook_paths
    )

    folder_dfg = os.path.join(OUTPUT_DIR, "execution_graphs")
    if not os.path.exists(folder_dfg):
        os.makedirs(folder_dfg)
    
    folder_code = os.path.join(OUTPUT_DIR, "code")
    if not os.path.exists(folder_code):
        os.makedirs(folder_code)
        
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
            
        name = '_'.join([filename.split('/')[3], filename.split('/')[4], filename.split('/')[-1].split('%')[-1]])
        name = name.replace("/", "_").replace(".", "_")
        
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
            
            if len(dfg._current_state.get_graph().nodes) < 2:
                raise Exception("Empty DFG")
                
            dfg.save_dfg(os.path.join(folder_dfg, f"{name}.execution_graph"))
            notebook.save_code(os.path.join(folder_code, f"{name}.code"))

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
    parser.add_argument("--start_date", default="2023-02-22", help="Oldest allowble date.")
    parser.add_argument("--end_date", default="2025-02-22", help="Newest allowble date.")
    parser.add_argument("--stop_index", default=1100000, help="End index")
    parser.add_argument("--parse_repos", action="store_true", help="If true, create the json file (notebook paths) by going through repositories")
    parser.add_argument("--notebook_paths", default=None, help="Path of json from github repo crawler")
    args = parser.parse_args()
    
    if args.parse_repos:
        repo_crawler = GitHubRepositoryCrawler(query = "", 
                                               last_acceptable_date=args.start_date,
                                               log_file="github_repo_crawler.log",
                                               filter_date_start=args.start_date,
                                               filter_date_end=args.end_date,
                                               save_folder=OUTPUT_DIR)
        repo_crawler.crawl()
        args.notebook_paths = repo_crawler.result_file

    mine_dataflows_on_github_dataset(args)