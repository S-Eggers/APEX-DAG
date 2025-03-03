import nbformat as nbf
import os
import time

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph


def notebook_out_of_cells(filename):
    notebook = nbf.v4.new_notebook()
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        python_code = '\n'.join(['\t'.join(line.split('\t')[1:]) for line in lines])
        
    code_cell = nbf.v4.new_code_cell(python_code)
    notebook.cells.append(code_cell)

    return notebook


if __name__ == "__main__":
    
    code_directory = '/home/eggers/data/apexdag_results/testing_jetbrains_dfg_100k_new/code_subset_attribute_error'
    execution_graphs_destination = '/home/eggers/data/apexdag_results/testing_jetbrains_dfg_100k_new/dfg_subset_attribute_error'
    
    if not os.path.exists(execution_graphs_destination):
        os.makedirs(execution_graphs_destination)
        
    notebook_objects = {}
    for filename in os.listdir(code_directory):
        name = filename.split('/')[-1].replace('.code', '')
        if filename.endswith(".code"):
            file_path = os.path.join(code_directory, filename)
            notebook_objects[name] = notebook_out_of_cells(file_path)
    
    # now we have all of the quasi-problematic objects
    for notebook_name, notebook_object in notebook_objects.items():
        notebook = Notebook(url=None, nb=notebook_object)
        notebook.create_execution_graph(greedy=True)
        execution_graph_end = time.time()
            
        dfg_start_time = time.time()
        dfg = DataFlowGraph()
        dfg.parse_notebook(notebook)
        dfg.optimize()
        dfg_end_time = time.time()
        
        # make sure destination exists
            
        dfg.save_dfg(os.path.join(execution_graphs_destination, f"{notebook_name}.execution_graph"))