import logging
import networkx as nx
from typing import List, Tuple
from ..core.types import PRType
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)

def add_to_stack(stack: list, item):
    if item not in stack:
        stack.append(item)

def extend_stack(stack: list, items):
    for item in items:
        add_to_stack(stack, item)

class AnnotationWIR:
    """
    Annotates the WIR using the Vamsa Annotation algorithm via the provided Knowledge Base.
    """
    def __init__(self, wir: nx.DiGraph, prs: List[PRType], knowledge_base):
        self.wir = wir
        self.prs = prs
        self.knowledge_base = knowledge_base
        self.annotated_wir = wir.copy()

    def _get_annotation(self, node):
        if node is None:
            return None
        annotations = self.annotated_wir.nodes[node].get("annotations", [])
        return annotations if annotations else None

    def annotate(self) -> nx.DiGraph:
        seed_set = self.find_import_nodes()

        for vs in seed_set:
            library, module = self.extract_library_and_module(vs)
            forward_stack = [vs]
            
            while forward_stack:
                node = forward_stack.pop(0)
                inputs, caller, process, outputs = self._extract_pr_elements(node)
                
                if self.check_if_visited(process):
                    continue
                self.visit_node(process)
                
                context = self._get_annotation(caller)
                if isinstance(context, list):
                    for subcontext in context:
                        input_annotations, annotations_output = self.knowledge_base(library, module, subcontext, process)
                        if input_annotations or annotations_output: break
                else:
                    input_annotations, annotations_output = self.knowledge_base(library, module, context, process)
                
                for vo, annotation_output in zip(outputs, annotations_output):
                    self._annotate_node(vo, annotation_output)

                annotated_inputs = []
                min_len = min(len(inputs), len(input_annotations))

                for input_node, input_annotation in zip(inputs[:min_len], input_annotations[:min_len]):
                    annotated_inputs.append(self._annotate_node(input_node, input_annotation))

                backward_stack = annotated_inputs

                while backward_stack:
                    previous_input, annotation = backward_stack.pop()
                    b_inputs, b_caller, b_process, b_outputs = self._extract_pr_elements(previous_input, node_type="output")

                    if self.check_if_visited(b_process): continue
                    self.visit_node(b_process)

                    outputs_annotations = [self._get_annotation(vo) for vo in b_outputs]
                    b_input_annotations = self.knowledge_base.back_query(outputs_annotations, b_process)

                    min_len_b = min(len(b_inputs), len(b_input_annotations))
                    for b_input_node, b_input_annotation in zip(b_inputs[:min_len_b], b_input_annotations[:min_len_b]):
                        self._annotate_node(b_input_node, b_input_annotation)
                        backward_stack.append((b_input_node, b_input_annotation))

                extend_stack(forward_stack, self.find_forward_prs((inputs, caller, process, outputs)))

        return self.annotated_wir

    def find_import_nodes(self):
        return [node for node in self.wir.nodes if "importas" in str(node).lower() or "importfrom" in str(node).lower()]

    def extract_library_and_module(self, node):
        library = None
        module = None
        for pred, _, edge_data in self.wir.in_edges(node, data=True):
            if edge_data.get("label") == "input":
                library = self.wir.nodes[pred].get("label", pred)
                break
        if library and "." in library:
            library, module = library.rsplit(".", 1)
        return library, module

    def _extract_pr_elements(self, node, node_type="operation"):
        inputs, context, outputs = [], None, []

        if node_type == "operation":
            for pred in self.wir.predecessors(node):
                edge_data = self.wir.get_edge_data(pred, node)
                if edge_data.get("label") == "input": inputs.append(pred)
                elif edge_data.get("label") == "caller": context = pred

            for succ in self.wir.successors(node):
                edge_data = self.wir.get_edge_data(node, succ)
                if edge_data.get("label") == "output": outputs.append(succ)

            return inputs, context, node, outputs

        elif node_type == "output":
            for pred in self.wir.predecessors(node):
                edge_data = self.wir.get_edge_data(pred, node)
                if edge_data.get("label") == "output":
                    return self._extract_pr_elements(pred, node_type="operation")
            self.visit_node(node)
            return [], None, None, [node]

    def _annotate_node(self, node, annotation):
        if "annotations" not in self.annotated_wir.nodes[node]:
            self.annotated_wir.nodes[node]["annotations"] = []
        if annotation not in self.annotated_wir.nodes[node]["annotations"]:
            self.annotated_wir.nodes[node]["annotations"].append(annotation)
        return (node, annotation)

    def check_if_visited(self, node) -> bool:
        if node is None: return True
        return self.annotated_wir.nodes[node].get("visited", False)

    def visit_node(self, node):
        self.annotated_wir.nodes[node]["visited"] = True

    def find_forward_prs(self, pr):
        outputs = pr[3]
        operation = pr[2]
        def is_connected(next_pr):
            next_pr_ids = (next_pr[0], next_pr[1], next_pr[2])
            return any(o in next_pr_ids for o in outputs) and next_pr[2] != operation
        return set([next_pr[2] for next_pr in self.prs if is_connected(next_pr)])