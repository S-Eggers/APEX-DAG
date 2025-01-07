import ast
import os
import random
import numpy as np

from ApexDAG.vamsa.utils import remove_comment_lines
from ApexDAG.vamsa.generate_wir import GenWIR
from ApexDAG.vamsa.annotate_wir import AnnotationWIR, KB
from ApexDAG import Notebook


random.seed(42)
np.random.seed(42)


def get_notebook(competition_path, notebook_name):
    notebook_path = os.path.join(competition_path, notebook_name)
    notebook = Notebook(notebook_path, cell_window_size=-1)
    notebook.create_execution_graph(greedy=True)
    code = notebook.code()
    code = remove_comment_lines(code) # get rid of the code comments (not necessary)
    return (code, notebook_name)

def traverse_competitions(competitions_path):
    competitions = os.listdir(competitions_path)
    for competition in competitions:
        if competition.endswith(".ipynb"):
            yield get_notebook(competitions_path, competition)

if __name__ == '__main__':
    
    competition_name = 'titanic'
    competition_path = "C:/Users/ismyn/UNI/TUB/notebook-dataset/notebooks/" + competition_name
    notebooks = traverse_competitions(competition_path)
    notebook_code, notebook_names = list(zip(*notebooks))

    # create data output directories in data/titanic_mvp_wir
    output_dir = os.path.join("data", f"{competition_name}_mvp_wir/")
    os.makedirs(output_dir, exist_ok=True)
    
    for notebook_name, code in zip(notebook_names, notebook_code):
        
        print(f"Generating WIR for {competition_name}/{notebook_name}")
        
        script_name = notebook_name.replace(".ipynb", "")
        output_path = os.path.join(output_dir, script_name)
        
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path,  'script.py'), "w") as f:
            f.write(code)
            
        parsed_ast = ast.parse(code)
        output_picture = os.path.join(output_path,  'wir.png')
        output_picture_annotated = os.path.join(output_path,  'annotated-wir.png')
        wir, prs, tuples = GenWIR(parsed_ast, output_filename=output_picture, if_draw_graph = True)
        annotated_wir = AnnotationWIR(wir, prs, KB(None))
        annotated_wir.annotate()
        input_nodes, output_nodes, caller_nodes, operation_nodes = tuples
        annotated_wir.draw_graph(input_nodes, output_nodes, caller_nodes, operation_nodes, output_picture_annotated)
            
        print("WIR written to", os.path.join(output_path,  'wir.png'))
        
    print("Notebooks written to", output_dir)
    
    
    