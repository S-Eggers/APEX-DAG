import os
import nbformat
from tqdm import tqdm
import logging

class LocalNotebookIterator:
    def __init__(self, local_path, log_file):
        self.local_path = local_path
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.filenames = self._find_notebooks()
        self.progress_bar = tqdm(total=len(self.filenames), desc="Processing Notebooks")
        self.current_index = 0

    def _find_notebooks(self):
        notebook_files = []
        for root, _, files in os.walk(self.local_path):
            for file in files:
                if file.endswith(".ipynb"):
                    notebook_files.append(os.path.join(root, file))
        return notebook_files

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.filenames):
            self.progress_bar.close()
            raise StopIteration

        filename = self.filenames[self.current_index]
        self.current_index += 1
        self.progress_bar.update(1)
        notebook = self._read_notebook(filename)

        if notebook is not None:
            return filename, notebook
        return self.__next__()

    def _read_notebook(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return nbformat.read(f, as_version=4)
        except Exception as e:
            self.logger.warning(f"Could not read notebook {filename}. Error: {e}")
            return None

    def print(self, filename="", message=""):
        message = str(message)
        self.logger.info(f"{filename}: {message}")
