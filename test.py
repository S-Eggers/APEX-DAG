import nbformat as nbf
import os
import time
import logging
from tqdm import tqdm

from ApexDAG.notebook import Notebook
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph as DataFlowGraph

# Set up logging to both file and terminal
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("notebook_errors.log"),
    logging.StreamHandler()
])

def notebook_out_of_cells(filename):
    notebook = nbf.v4.new_notebook()
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        python_code = '\n'.join(['\t'.join(line.split('\t')[1:]) for line in lines])
        
    code_cell = nbf.v4.new_code_cell(python_code)
    notebook.cells.append(code_cell)

    return notebook

if __name__ == "__main__":
    code_directory = '/home/eggers/data/apexdag_results/testing_jetbrains_dfg_100k_new/code_subset_attribute_error_new_batch'
    execution_graphs_destination = '/home/eggers/data/apexdag_results/testing_jetbrains_dfg_100k_new/dfg_subset_attribute_error_new_batch'
    draw = True
    draw_destination = os.path.join(execution_graphs_destination, 'draw')
    
    if not os.path.exists(execution_graphs_destination):
        os.makedirs(execution_graphs_destination)
    if draw and not os.path.exists(draw_destination):
        os.makedirs(draw_destination)
        
    notebook_objects = {}
    for filename in os.listdir(code_directory):
        name = filename.split('/')[-1].replace('.code', '')
        if filename.endswith(".code"):
            file_path = os.path.join(code_directory, filename)
            notebook_objects[name] = notebook_out_of_cells(file_path)
    
    # now we have all of the quasi-problematic objects
    for notebook_name, notebook_object in tqdm(notebook_objects.items(), desc="Processing Notebooks"):
        try:
            notebook = Notebook(url=None, nb=notebook_object)
            notebook.create_execution_graph(greedy=True)
            execution_graph_end = time.time()
                
            dfg_start_time = time.time()
            dfg = DataFlowGraph()
            dfg.parse_notebook(notebook)
            dfg.optimize()
            dfg_end_time = time.time()
            
            # make sure destination exists
            if draw:
                dfg.draw(os.path.join(draw_destination, f"{notebook_name}.png"))
                
            dfg.save_dfg(os.path.join(execution_graphs_destination, f"{notebook_name}.execution_graph"))
        
        except Exception as e:
            error_message = f"Error processing notebook {notebook_name}: {str(e)}"
            logging.error(error_message)
            with open("notebook_errors.txt", "a") as error_file:
                error_file.write(error_message + "\n")