from tqdm import tqdm
import os
import json
import logging
import time

from ApexDAG.scripts.crawl_notebooks.notebook_processor import NotebookProcessor
from ApexDAG.scripts.crawl_notebooks.github_crawler.github_repository_notebook_iterator import PaginatedNotebookIterator

class NotebookMiner:
    def __init__(self, iterator, save_dir, log_file, start_index=0, stop_index=None):
        self.save_dir = save_dir
        self.logs_dir = os.path.join(save_dir, 'logs')
        
        self.logger = logging.getLogger(f'NotebookMiner ({log_file})')
        self.logger.setLevel(logging.INFO)

        os.makedirs(self.logs_dir, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(self.logs_dir, log_file))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.iterator = iterator
        self.processor = NotebookProcessor(save_dir, log_file)

    def download_and_mine_notebooks(self, output_file_name='annotated_test.json', delay=0):
        """
        Download and process Jupyter notebooks using the Notebook Iterator.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        all_annotations = {}
        for filename, notebook in tqdm(self.iterator, desc="Mining Notebooks"):
            try:
                if notebook.get("notebook_content", None) is not None: # github crealer
                    notebook = notebook["notebook_content"]
                if notebook is None:
                    continue
                    
                code = self.processor.extract_code(notebook)
                annotations_object = self.processor.process_cells(code)
                all_annotations[filename] = annotations_object

                if delay:
                    time.sleep(delay)

            except Exception as e:
                self.logger.warning(f"Error processing notebook {filename}: {e}")
                continue

        # Save the mined annotations to a JSON file
        output_file_path = os.path.join(self.save_dir, output_file_name)
        try:
            with open(output_file_path, 'w') as f:
                json.dump(all_annotations, f, indent=4)
            self.logger.info(f"Annotations successfully saved to {output_file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save annotations: {e}")



if __name__ == "__main__":
    SAVE_DIR = "data/notebooks/github"
    GITHUB_API_URL = "https://api.github.com/search/code"
    START_INDEX = 50
    STOP_INDEX = 80

    miner = NotebookMiner(
        iterator = PaginatedNotebookIterator(
        query="extension:ipynb",
        per_page=99,
        max_results=100,
        search_url=GITHUB_API_URL,
        log_file=f'notebook_miner_{START_INDEX}_{STOP_INDEX}.log'
    ),
        save_dir=SAVE_DIR,
        log_file=f'notebook_miner_{START_INDEX}_{STOP_INDEX}.log',
        start_index=START_INDEX,
        stop_index=STOP_INDEX
    )

    miner.download_and_mine_notebooks(output_file_name="annotated_notebooks.json")
