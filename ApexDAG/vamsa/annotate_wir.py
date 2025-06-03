import ast
from ApexDAG.vamsa.generate_wir import GenWIR
from ApexDAG.vamsa.utils import remove_id
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def add_to_stack(stack, item):
    if item not in stack:
        stack.append(item)
        
def extend_stack(stack, items):
    for item in items:
        add_to_stack(stack, item)
               
class KB:
    def __init__(self, knowledge_base = None):
        self.knowledge_base = pd.DataFrame([
        {"Library": "catboost", "Module": None, "Caller": None, "API Name": "CatBoostClassifier", "Inputs": ["hyperparameter"], "Outputs": ["model"]},
        {"Library": "catboost", "Module": None, "Caller": "model", "API Name": "fit", "Inputs": ['features', 'labels'], "Outputs": ["trained model"]},
        {"Library": "sklearn", "Module": "model.selection", "Caller": None, "API Name": "train_test_split", "Inputs": ["features", "labels"], "Outputs": ["features", "validation features", "labels", "validation labels"]},
        {"Library": "pandas", "Module": None, "Caller": None, "API Name": "read_csv", "Inputs": ["file_path"], "Outputs": ["data"]},
        {"Library": "pandas", "Module": None, "Caller": "data", "API Name": "iloc", "Inputs": ["columns_range"], "Outputs": ["features"]},
        {"Library": "pandas", "Module": None, "Caller": "data", "API Name": "drop", "Inputs": ["features"], "Outputs": ["features"]},
        {"Library": "pandas", "Module": None, "Caller": "data", "API Name": "filter", "Inputs": ["condition"], "Outputs": ["features"]},
        {"Library": "sklearn", "Module": "preprocessing", "Caller": None, "API Name": "StandardScaler", "Inputs": ["features"], "Outputs": ["features"]},
        {"Library": "sklearn", "Module": "preprocessing", "Caller": None, "API Name": "LabelEncoder", "Inputs": ["labels"], "Outputs": ["labels"]},
        {"Library": "sklearn", "Module": "linear_model", "Caller": None, "API Name": "LogisticRegression", "Inputs": ["features", "labels"], "Outputs": ["model"]},
        {"Library": "sklearn", "Module": "linear_model", "Caller": "model", "API Name": "fit", "Inputs": ["train_features", "train_labels"], "Outputs": ["trained_model"]},
        {"Library": "sklearn", "Module": "linear_model", "Caller": "trained_model", "API Name": "predict", "Inputs": ["features"], "Outputs": ["predictions"]},
        {"Library": "tensorflow", "Module": "keras", "Caller": None, "API Name": "Sequential", "Inputs": ["layers"], "Outputs": ["model"]},
        {"Library": "tensorflow", "Module": "keras", "Caller": "model", "API Name": "compile", "Inputs": ["optimizer", "loss_function", "metrics"], "Outputs": ["model"]},
        {"Library": "tensorflow", "Module": "keras", "Caller": "compiled_model", "API Name": "fit", "Inputs": ["features", "labels"], "Outputs": ["trained_model"]},
        {"Library": "tensorflow", "Module": "keras", "Caller": "trained_model", "API Name": "predict", "Inputs": ["test_features"], "Outputs": ["predictions"]}
    ])
            
        self.knowledge_base_traversal = pd.DataFrame([
        {"Library": None, "Module": None, "Caller": 'data', "API Name": "Subscript", "Inputs": ["selected columns"]},
        {"Library": None, "Module": None, "Caller": 'data', "API Name": "drop", "Inputs": ["dropped columns"]}])
            

    def __call__(self, L, L_prime, c, p):
        # Filter knowledge base based on parameters
        filtered_kb = self.knowledge_base

        if L is not None:
            filtered_kb = filtered_kb[(filtered_kb['Library'].fillna('') == remove_id(L))]
        if (not filtered_kb.empty) and L_prime is not None:
            filtered_kb = filtered_kb[(filtered_kb['Module'].fillna('') == remove_id(L_prime))]
        
        if (not filtered_kb.empty) and c is not None:
            filtered_kb = filtered_kb[(filtered_kb['Caller'] == remove_id(c))]
        elif (not filtered_kb.empty):
            filtered_kb = filtered_kb[(filtered_kb['Caller'].isna())]

        if (not filtered_kb.empty) and p is not None:
            filtered_kb = filtered_kb[(filtered_kb['API Name'].fillna('') == remove_id(p))]
        elif (not filtered_kb.empty):
            filtered_kb = filtered_kb[(filtered_kb['API Name'].isna())]

        if len(filtered_kb) > 1:
            raise ValueError("Knowledge base contains multiple matching rows!")

        inputs = filtered_kb["Inputs"].values[0] if len(filtered_kb["Inputs"].values) == 1 else []
        outputs = filtered_kb["Outputs"].values[0] if len(filtered_kb["Outputs"].values) == 1 else []

        return inputs, outputs
    
    def back_query(self, O, p): # TODO: make the name better
        # Filter knowledge base based on parameters
        def has_similar_elements(row_list, provided_list):
            min_len = min(len(row_list), len(provided_list))
            return all(p == r or p is None for p, r in zip(provided_list[:min_len], row_list[:min_len]))

        filtered_kb = self.knowledge_base

        if p is not None:
            filtered_kb = filtered_kb[(filtered_kb['API Name'].fillna('') == remove_id(p))]
        if (not filtered_kb.empty) and O is not None:
            filtered_kb = filtered_kb[(filtered_kb['Outputs'].apply(lambda x: has_similar_elements(x, O)))]
        
        if len(filtered_kb) > 1:
            raise ValueError("Knowledge base contains multiple matching rows!")
        inputs = filtered_kb["Inputs"].values[0] if len(filtered_kb["Inputs"].values) == 1 else []
        
        return inputs 
    
    
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
    
    def _get_annotation(self, node):
        """Get single annotation of a node."""
        if node is None:
            return None
        annotations = self.annotated_wir.nodes[node].get('annotations', [])
        # if more than 1 raise warning
        if len(annotations) > 1:
            raise ValueError(f"Node {node} has multiple annotations: {annotations}")
        return annotations[0] if annotations else None

    def annotate(self):
        """Annotates the WIR using the PRS Algorithm Annotation."""
        # Find seed
        seed_set = self.find_import_nodes()
        
        for vs in seed_set:
            library, module = self.extract_library_and_module(vs)

            # DFS
            forward_stack = [vs]
            while forward_stack:
                node = forward_stack.pop()  # pop a process relation
                pr = inputs, caller, process, outputs = self._extract_pr_elements(node)  # extract PR = <I, c, p, O>
                
                if self.check_if_visited( process):
                    continue
                self.visit_node(process)
                context = self._get_annotation(caller) # get annotation of caller

                input_annotations, annotations_output = self.knowledge_base(library, module, context, process)
                
                for vo, annotation_output in zip(outputs, annotations_output):
                    self._annotate_node(vo, annotation_output)
                
                annotated_inputs = []
                min_len = min(len(inputs), len(input_annotations))
                
                for input_node, input_annotation in zip(inputs[:min_len], input_annotations[:min_len]):
                    annotated_inputs.append(self._annotate_node(input_node, input_annotation))

                backward_stack = annotated_inputs # Start from input nodes (I)

                while backward_stack:
                    previous_input, annotation = backward_stack.pop()  # Pop a process relation
                    #  Annotate inputs using knowledge base
                    inputs, caller, process, outputs = self._extract_pr_elements(previous_input, node_type = 'output')  # Extract PR = <I, c, p, O> - find to which inuts where outputs
                    
                    if self.check_if_visited( process):
                        continue
                    self.visit_node(process)
                    
                    outputs_annotations = [self._get_annotation(vo) for vo in outputs] # get the annotation of the previous input and other outputs of same operation
                    input_annotations = self.knowledge_base.back_query(outputs_annotations, process)

                    min_len = min(len(inputs), len(input_annotations))
                    for input_node, input_annotation in zip(inputs[:min_len], input_annotations[:min_len]):
                        self._annotate_node(input_node, input_annotation)
                        backward_stack.append((input_node, input_annotation))

                extend_stack(forward_stack, self.find_forward_prs(pr))

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

    def _extract_pr_elements(self, node, node_type='operation'):
        """
        Extract elements of a process relation PR = <I, c, p, O> for a given process node.

        :param node: The process node from the WIR graph.
        :return: A tuple (inputs, context, process, outputs).
        """
        inputs = []
        context = None
        outputs = []
        
        if node_type == 'operation':
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
        
        elif node_type == 'output':
            for pred in self.wir.predecessors(node):
                edge_data = self.wir.get_edge_data(pred, node)
                if edge_data.get("edge_type") == "operation_to_output":
                    operation = pred # find operation and do propper pr extraction
                    return self._extract_pr_elements(operation, node_type='operation')
            self.visit_node(node)
            return None, None, None, node
                    

    def _annotate_node(self, node, annotation):
        """Add an annotation to a node in the WIR."""
        if 'annotations' not in self.annotated_wir.nodes[node]:
            self.annotated_wir.nodes[node]['annotations'] = []
        self.annotated_wir.nodes[node]['annotations'].append(annotation)
        return (node, annotation)
        
    def check_if_visited(self, node):
        if node is None:
            return True # this means we have an output node only and do not want to traverse it either way
        if 'visited' in self.annotated_wir.nodes[node]:
            return self.annotated_wir.nodes[node]['visited']
        return False
    def visit_node(self, node):
        self.annotated_wir.nodes[node]['visited'] = True

    def find_forward_prs(self, pr):
        """Find forward PRs connected to the outputs of a given PR."""
        outputs = pr[3]
        operation = pr[2]
        
        def is_connected(next_pr):
            next_pr_ids = (next_pr[0], next_pr[1], next_pr[2])
            return any(o in next_pr_ids for o in outputs) and next_pr[2] != operation
        
        return set([next_pr[2] for next_pr in self.prs if is_connected(next_pr)]) # get operations connected
    
    def draw_graph(self, input_nodes, output_nodes, caller_nodes, operation_nodes, output_filename):
    
        labels = {node: remove_id(node) + f' :Annotation {self._get_annotation(node)}' for node in self.annotated_wir.nodes()}
        
        plt.figure(figsize=(200, 40))
        pos = graphviz_layout(self.annotated_wir, prog='dot')

        nx.draw_networkx_nodes(self.annotated_wir, pos, nodelist=input_nodes, node_shape='o') 
        nx.draw_networkx_nodes(self.annotated_wir, pos, nodelist=caller_nodes, node_shape='o')
        nx.draw_networkx_nodes(self.annotated_wir, pos, nodelist=operation_nodes, node_shape='s') 
        nx.draw_networkx_nodes(self.annotated_wir, pos, nodelist=output_nodes, node_shape='o')

        edges = self.annotated_wir.edges(data=True)
        edge_colors = [d['color'] for (_, _, d) in edges]
        
        nx.draw_networkx_edges(self.annotated_wir, pos, edgelist=edges, edge_color=edge_colors, arrows=True)
        nx.draw_networkx_labels(self.annotated_wir, pos, labels=labels)
        
        
        plt.legend()
        plt.savefig(output_filename)
        plt.close()

if __name__ == "__main__":
    file_path = f'data/raw/test_vamsa.py'
    location_related_attributes = ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']

    with open(file_path, 'r') as file:
        file_content = file.read()
    file_lines = file_content.split('\n')
        
    parsed_ast = ast.parse(file_content)
    wir, prs, tuples = GenWIR(parsed_ast, output_filename=f'output/wir-unannotated-test.png', if_draw_graph = True)
    annotated_wir = AnnotationWIR(wir, prs, KB(None))
    annotated_wir.annotate()
    input_nodes, output_nodes, caller_nodes, operation_nodes = tuples
    annotated_wir.draw_graph(input_nodes, output_nodes, caller_nodes, operation_nodes, 'output/annotated-wir-final.png')
