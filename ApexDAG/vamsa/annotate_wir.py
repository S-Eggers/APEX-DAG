import ast
from ApexDAG.vamsa.generate_wir import GenWIR
from ApexDAG.vamsa.utils import remove_id
import networkx as nx

class KB:
    def __init__(self, knowledge_base = None):
        import pandas as pd
        self.knowledge_base = pd.DataFrame([
    {"Library": "catboost", "Module": None, "Caller": None, "API Name": "CatBoostClassifier", "Inputs": ["eval metrics: hyperparameter"], "Outputs": ["model"]},
    {"Library": "catboost", "Module": None, "Caller": "model", "API Name": "fit", "Inputs": ['features', 'labels'], "Outputs": ["trained model"]},
    {"Library": "sklearn", "Module": "model selection", "Caller": None, "API Name": "train_test_split", "Inputs": ["features", "labels"], "Outputs": ["features", "validation features"]},])
        

    def __call__(self, L, L_prime, c, p):
        # Filter knowledge base based on parameters
        filtered_kb = self.knowledge_base

        if L is not None:
            filtered_kb = filtered_kb[(filtered_kb['Library'].fillna('') == remove_id(L))]
        if L_prime is not None:
            filtered_kb = filtered_kb[(filtered_kb['Module'].fillna('') == remove_id(L_prime))]
        
        filtered_kb = filtered_kb[(filtered_kb['Caller'].fillna('') == remove_id(c))]
        filtered_kb = filtered_kb[(filtered_kb['API Name'].fillna('') == remove_id(p))]

        return filtered_kb["Inputs"], filtered_kb["Outputs"]
    
    
class AnnotationWIR:
    def __init__(self, wir, prs, knowledge_base):
        """
        Initialize the WIR annotation class.
        
        :param wir: Whole Intermediate Representation (WIR), graph-like data structure.
        :param prs: Process relations (list of PR elements <I, c, p, O>).
        :param knowledge_base: A function KB(L, L′, c, p) or KB(O, p) that provides annotations.
        """
        self.wir = wir
        self.prs = prs
        self.knowledge_base = knowledge_base
        self.annotated_wir = wir.copy()  # Start with a copy for annotations

    def annotate(self):
        """Annotates the WIR using the PRS Algorithm Annotation."""
        # Find seed
        seed_set = self.find_import_nodes()
        
        for vs in seed_set:
            library, module = self.extract_library_and_module(vs)

            # DFS
            forward_stack = [vs]
            while forward_stack:
                node = forward_stack.pop()  # Pop a process relation
                pr = inputs, context, process, outputs = self.extract_pr_elements(node)  # Extract PR = <I, c, p, O>

                #  Annotate in- and outputs using the mock knowledge base
                annotation_input, annotation_output = self.knowledge_base(library, module, context, process)
                for index_o, vo in enumerate(outputs):
                    if annotation_output is not None and len(annotation_output) > index_o:
                        self.annotate_node(vo, annotation_output[index_o])
                
                annotated_inputs = []
                for index_i, vi in enumerate(inputs):
                    if annotation_input is not None and len(annotation_input) > index_i: # FIXME: later not allowing for kward different order
                        annotated_inputs.append(self.annotate_node(vi, annotation_input[index_i]))

                backward_stack = annotated_inputs # Start from input nodes (I)

                while backward_stack:
                    pr, annotation = backward_stack.pop()  # Pop a process relation
                    inputs, context, process, outputs = self.extract_pr_elements(pr)

                    #  Annotate inputs using knowledge base
                    for vi in inputs:
                        annotation = self.knowledge_base(outputs, process)
                        self.annotate_node(vi, annotation)

                    # Add unvisited PRs reachable from inputs to the stack
                    backward_stack.extend(self.find_backward_prs(pr))
                forward_stack.extend(self.find_forward_prs(pr))

        return self.annotated_wir

    def find_import_nodes(self):
        """Identify all seed nodes in WIR"""
        return [node for node in self.wir.nodes if 'importas' in node.lower() or 'importfrom' in node.lower()]

    def extract_library_and_module(self, node):
        """
        Extract the library (L) and module (L') from an import node in a nx.DiGraph.
        
        :param node: The starting import process node.
        :return: A tuple (library, module) where L is the library, and L' is the module.
        """
        library = None
        module = None
        # Traverse incoming edges to find the library (input_to_operation edges)
        for pred, _, edge_data in self.wir.in_edges(node, data=True):
            if edge_data.get('edge_type') == 'input_to_operation':
                library = self.wir.nodes[pred]['label']  # Get library name
                break
        # If there is a dot, there are also submodules
        if library and '.' in library:
            library, module = library.rsplit('.', 1)

        return library, module

    def extract_pr_elements(self, node):
        """
        Extract elements of a process relation PR = <I, c, p, O> for a given process node.

        :param node: The process node from the WIR graph.
        :return: A tuple (inputs, context, process, outputs).
        """
        inputs = []
        context = None
        outputs = []
        
        for pred in self.wir.predecessors(node):
            edge_data = self.wir.get_edge_data(pred, node)
            if edge_data.get("edge_type") == "input_to_operation":
                inputs.append(pred)  # add input nodes
            elif edge_data.get("edge_type") == "caller_to_operation":
                context = pred  # add caller node

        for succ in self.wir.successors(node):
            edge_data = self.wir.get_edge_data(node, succ)
            if edge_data.get("edge_type") == "operation_to_output":
                outputs.append(succ)  # add output nodes

        operation = node 
        return inputs, context, operation, outputs

    def annotate_node(self, node, annotation):
        """Add an annotation to a node in the WIR."""
        if 'annotations' not in self.annotated_wir.nodes[node]:
            self.annotated_wir.nodes[node]['annotations'] = []
        self.annotated_wir.nodes[node]['annotations'].append(annotation)
        return (node, annotation)
        

    def find_forward_prs(self, pr):
        """Find forward PRs connected to the outputs of a given PR."""
        outputs = pr[3]
        operation = pr[2]
        
        def is_connected(next_pr):
            next_pr_ids = (next_pr[0], next_pr[1], next_pr[2])
            return any(o in next_pr_ids for o in outputs) and next_pr[2] != operation
        
        return set([next_pr[2] for next_pr in self.prs if is_connected(next_pr)]) # get operations connected
    
    def find_backward_prs(self, pr):
        """Find backward PRs connected to the inputs of a given PR."""
        pass


if __name__ == "__main__":
    file_path = f'data/raw/test_vamsa.py'
    location_related_attributes = ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']

    with open(file_path, 'r') as file:
        file_content = file.read()
    file_lines = file_content.split('\n')
        
    parsed_ast = ast.parse(file_content)
    wir, prs = GenWIR(parsed_ast, output_filename=f'output/wir-annotated-test.png', if_draw_graph = True)
    annotated_wir = AnnotationWIR(wir, prs, KB(None))
    annotated_wir.annotate()
