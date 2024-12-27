import os
import sys
import pandas as pd

from scripts.notebook import Notebook
from scripts.util.notebook_stat_miner import NotebookStatMiner
from scripts.util.kaggle_dataset_iterator import KaggleDatasetIterator


def analyze_kaggle_dataset(args, logger):
    column_names = ["name", "url", "notebook", "type", "evaluation_metric", "selection_criteria", "beginner", "prize", "imports", "import_usage", "import_counts",  "cells_with_imports", "custom_classes", "custom_functions", "num_code_cells"]
    dataset_stats = pd.DataFrame(columns=column_names)
    kaggle_iterator = KaggleDatasetIterator(os.path.join(os.getcwd(), "data", "raw", "notebooks"))
    for competition in kaggle_iterator:
        for notebook_file in competition["ipynb_files"]:
            try:
                notebook_path = os.path.join(competition["subfolder_path"], notebook_file)
                notebook = Notebook(notebook_path)
                notebook.create_execution_graph(greedy=args.greedy)
                results = NotebookStatMiner(notebook).mine()

            except:
                print(f"Error processing notebook {notebook_path}")
                sys.exit(-1)
                
            notebook_stats = {
                "name": competition["json_file"]["name"],
                "url": competition["json_file"]["url"],
                "notebook": os.path.join(os.path.basename(os.path.dirname(notebook_path)), notebook_file),
                "type": competition["json_file"]["type"].split(","),
                "evaluation_metric": competition["json_file"]["evaluationMetric"],
                "selection_criteria": competition["json_file"]["selectionCriteria"].split(","),
                "beginner": bool(competition["json_file"]["beginner"]),
                "prize": competition["json_file"]["prize"],
                "imports": results[0],
                "import_usage": results[1],
                "import_counts": results[2],
                "cells_with_imports": results[5],
                "custom_classes": results[3],
                "custom_functions": results[4],
                "num_code_cells": notebook.count_code_cells()
            }
            stats_df = pd.DataFrame([notebook_stats], columns=column_names)
            dataset_stats = pd.concat([dataset_stats, stats_df], ignore_index=True)
        
    dataset_stats.to_csv(os.path.join(os.getcwd(), "output", "kaggle_dataset_stats.csv"), index=False)