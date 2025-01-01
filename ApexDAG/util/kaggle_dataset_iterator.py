import os
import json
from tqdm import tqdm

class KaggleDatasetIterator:
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.results = []
        self._iterator: tqdm = None
        
    def __iter__(self):
        self._process_folders()
        self._iterator = tqdm(self.results, desc="Processed competitions")
        for result in self._iterator:
            yield result
            
    def print(self, message: str):
        if self._iterator:
            self._iterator.write(message)
        else:
            raise RuntimeError("No iterator initialized") 

    def _process_folders(self):
        # Iterate through all items in the main folder
        for subfolder in os.listdir(self.main_folder):
            subfolder_path = os.path.join(self.main_folder, subfolder)

            # Check if the item is a directory
            if os.path.isdir(subfolder_path):
                json_file = None
                ipynb_files = []

                # Iterate through all items in the subfolder
                for item in os.listdir(subfolder_path):
                    item_path = os.path.join(subfolder_path, item)

                    # Check if the item is a JSON file
                    if item.endswith(".json"):
                        with open(item_path, "r", encoding="utf-8") as f:
                            json_file = json.load(f)

                    # Check if the item is an IPython Notebook file
                    elif item.endswith(".ipynb"):
                        ipynb_files.append(item)

                self.results.append({
                    "subfolder": subfolder,
                    "subfolder_path": subfolder_path,
                    "json_file": json_file,
                    "ipynb_files": ipynb_files
                })
