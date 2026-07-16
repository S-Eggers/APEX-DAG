import json
import os

from tqdm import tqdm

class KaggleDatasetIterator:
    def __init__(self, main_folder: str) -> None:
        self.main_folder = main_folder
        self.results = []
        self._iterator: tqdm = None

    def __iter__(self) -> object:
        self._process_folders()
        self._iterator = tqdm(self.results, desc="Processed competitions")
        yield from self._iterator

    def print(self, message: str) -> None:
        if self._iterator:
            self._iterator.write(message)
        else:
            raise RuntimeError("No iterator initialized")

    def _process_folders(self) -> None:
        for subfolder in os.listdir(self.main_folder):
            subfolder_path = os.path.join(self.main_folder, subfolder)

            if os.path.isdir(subfolder_path):
                json_file = None
                ipynb_files = []

                for item in os.listdir(subfolder_path):
                    item_path = os.path.join(subfolder_path, item)

                    if item.endswith(".json"):
                        with open(item_path, encoding="utf-8") as f:
                            json_file = json.load(f)

                    elif item.endswith(".ipynb"):
                        ipynb_files.append(item)

                self.results.append(
                    {
                        "subfolder": subfolder,
                        "subfolder_path": subfolder_path,
                        "json_file": json_file,
                        "ipynb_files": ipynb_files,
                    }
                )
