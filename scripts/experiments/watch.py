import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from scripts.notebook import Notebook
from scripts.ast.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph


class NotebookHandler(FileSystemEventHandler):
    def __init__(self, file_to_watch, logger, action):
        super().__init__()
        self.file_to_watch = file_to_watch
        self.logger = logger
        self.action = action

    def on_modified(self, event):
        # Überprüfen, ob die geänderte Datei die überwachte Datei ist
        if event.src_path.endswith(self.file_to_watch):
            self.logger.info(f"Change in {self.file_to_watch} detected. Generating new dataflow graph.")
            start_time = time.time()
            self.action(event.src_path)
            end_time = time.time()
            self.logger.info(f"Building, optimizing, and drawing dataflow graph took {end_time - start_time}s")


def execute_action(file_path):
    Notebook.VERBOSE = True
    DataFlowGraph.VERBOSE = False

    notebook_path = os.path.join(os.getcwd(), "data", "raw", file_path)
    notebook = Notebook(notebook_path)
    notebook.create_execution_graph(greedy=True)

    dfg = DataFlowGraph()
    dfg.parse_notebook(notebook)
    dfg.optimize()
    dfg.draw()

def watch(args, logger):
    event_handler = NotebookHandler(args.notebook, logger, execute_action)
    path_to_watch = "."
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)

    try:
        observer.start()
        logger.info(f"Watching for changes in {args.notebook}...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
