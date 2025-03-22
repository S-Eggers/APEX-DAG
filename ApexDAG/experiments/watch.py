import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph


class NotebookHandler(FileSystemEventHandler):
    def __init__(self, file_to_watch, logger, action):
        super().__init__()
        self.file_to_watch = os.path.basename(file_to_watch)
        self.logger = logger
        self.action = action
        self.last_file_change = 0

    def on_modified(self, event):
        if event.src_path.endswith(self.file_to_watch):
            file_mod_time = os.path.getmtime(event.src_path)
            if not hasattr(self, 'last_file_change') or file_mod_time > self.last_file_change:
                self.logger.info(f"Change in {self.file_to_watch} detected. Generating new dataflow graph.")
                start_time = time.time()
                self.action(event.src_path, self.logger)
                end_time = time.time()
                self.logger.info(f"Building, optimizing, and drawing dataflow graph took {end_time - start_time}s")
                self.last_file_change = file_mod_time


def execute_action(file_path: str, logger: logging.Logger) -> None:
    Notebook.VERBOSE = False
    DataFlowGraph.VERBOSE = False

    notebook = Notebook(file_path)
    notebook.create_execution_graph(greedy=True)

    dfg = DataFlowGraph()
    start_time = time.time()
    dfg.parse_notebook(notebook)
    end_time = time.time()
    logger.info(f"Generating dataflow took {end_time - start_time}s")
    start_time = time.time()
    dfg.optimize()
    end_time = time.time()
    logger.info(f"Optimizing dataflow graph took {end_time - start_time}s")
    start_time = time.time()
    path = os.path.join(os.getcwd(), "apex-dag-jupyter", "src", "app", "data_flow_graph")
    dfg.webrender(path)
    end_time = time.time()
    logger.info(f"Drawing dataflow graph took {end_time - start_time}s")

def watch(args, logger: logging.Logger) -> None:
    file_path = args.notebook
    event_handler = NotebookHandler(file_path, logger, execute_action)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(file_path), recursive=False)

    try:
        observer.start()
        logger.info(f"Watching for changes in {file_path}...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
