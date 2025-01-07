import itertools
import ast
import logging
import numpy as np 
import random 

from ast import iter_child_nodes
from matplotlib import pyplot as plt
from typing import Set, Tuple, Optional

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from ApexDAG.vamsa.utils import (
    add_id,
    remove_id,
    is_empty_or_none_list,
    flatten,
    check_bipartie,
    merge_prs,
    WIRNodeType,
    PRType,
    WIRNode
)

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.WARNING)  # Adjust for debugging
logger = logging.getLogger(__name__)  # Get a logger for this module

def extract_from_node(node, field)-> Optional[WIRNodeType]: # main function, not defined in the paper
    """Extracts information from a node."""
    node = node.node
    if node is None:
        return WIRNode(None)
    
    match node.__class__.__name__:
        case "Assign":
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
            elif field == "input":
                return WIRNode(node.value)
            elif field == "output":
                return WIRNode(node.targets) # also needs id attribute
        case "Call":
            if field == "operation":
                return WIRNode(node.func, True)
            elif field == "input":
                return WIRNode(node.args + node.keywords)
            elif node.func and hasattr(node.func, 'id') and isinstance(node.func.id, str) and field == "output":
                return WIRNode(node.func.id + add_id())
        case "Attribute":
            if field == "caller":
                return WIRNode(node.value)
            elif field == "operation":
                return WIRNode(node.attr, True)
            elif field == "output":
                return WIRNode(node.attr + add_id()) # also needs id attribute
        case "Name":
            if field == "output":
                return WIRNode(node.id) # + add_id() # also needs id attribute, does not work sadly...
        case "Constant":
            if field == "output":
                return WIRNode(f"{node.value}{add_id()}") # also needs id attribute
        case 'Import':
            if field == "output":
                return WIRNode(node.names)
        case 'Module':
            return WIRNode(None)
        case 'alias':
            if field == "output":
                return WIRNode(node.asname if node.asname is not None else node.name) # also needs id attribute
            elif field == "input":
                return WIRNode(node.name if node.asname is not None else node.name + add_id())
            if field == "operation":
                return WIRNode("ImportAs" + add_id()) # node.__class__.__name__ + add_id() # name change to make it same an in paper
        case 'ImportFrom': # this causes not bipartie
            if field == "operation":
                return WIRNode(node.__class__.__name__  + add_id())
            elif field == "input":
                return WIRNode(node.module)
            elif field == "output": # change it to get bipartie graph
                return WIRNode([child.name if child.asname is None else child for child in node.names]) # omit alias if not necesary
        case 'Store':
            pass # TODO
        case 'Subscript':
            if field == "operation":
                return WIRNode(node.__class__.__name__  + add_id())
            elif field == "input":
                return WIRNode(node.slice)
            elif field == "caller":
                return WIRNode(node.value)
        case 'Tuple': # this omits the modelling of tuple...
            if field == "output":
                return WIRNode(node.elts)
        case 'Slice':
            if field == "operation":
                return WIRNode(node.__class__.__name__  + add_id())
            elif field == "input":
                return WIRNode([value if value is not None else '' + add_id() for value in (node.lower, node.upper, node.step)])
        case 'List':
            if field == "input":
                return WIRNode(node.elts)
            elif field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Expr': # TODO
            if field == 'output':
                return WIRNode(node.value)
        case 'For': # target, iter, body, orelse
            if field == "input":
                return WIRNode(node.iter)
            elif field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
            elif field == "output":
                return WIRNode(node.target) # not too sure about this
        case 'Compare': # left, ops, comparators
            if field == "input":
                return WIRNode([node.left] + node.comparators)
            elif field == "operation":
                return WIRNode(node.ops)
        case 'BinOp':
            if field == "input":
                return WIRNode([node.left, node.right])
            elif field == "operation":
                return WIRNode(node.op)
        case 'Lambda': #
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'FunctionDef': #
            pass
        case 'keyword':
            if field == "input":
                return WIRNode(node.value)
            elif field == "output":
                if node.arg is not None:
                    return WIRNode(node.arg + add_id())
            elif field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Add':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Sub':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Mult':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Div':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Eq':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Lt':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Gt':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'GtE':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'LtE':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'BitAnd':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'Mod':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'ListComp':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
            elif field == "input":
                return WIRNode(node.generators + [node.elt])
        case 'UnaryOp':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
            elif field == "input":
                return WIRNode(node.operand)
        case 'comprehension': 
            if field == "input":
                return WIRNode(node.iter)
            elif field == "output":
                return WIRNode(node.target)
            elif field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case 'IfExp':
            if field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
            if field == "input":
                return WIRNode([node.test,node.test, node.orelse])
        case 'Dict':
            if field == "input":
                return WIRNode([ast.Tuple([key,value]) for key, value in zip(node.keys, node.values)])
            elif field == "operation":
                return WIRNode(node.__class__.__name__ + add_id())
        case _:
            logger.warning(f"Field {field} not found in node {node.__class__.__name__}")  
    logger.info(f"Field {field} not found in node {node.__class__.__name__}")  
    return WIRNode(None)


def GenPR(v: WIRNode, PRs: list[PRType]) -> Tuple[WIRNodeType, list[PRType]]:
    """
    Processes a single AST node to generate WIR variables and update PRs.
    
    :param v: AST node.
    :param PRs: List of PRs generated so far.
    :return: Tuple containing a set of WIR variables and updated PRs.
    """
    
    isAttribute = v.isAttribute
    
    if v is None:
        return WIRNode(None), PRs
    
    if isinstance(v.node, list): # handles input/ output lists
        os = []
        for node in v.node:
            o, PRs = GenPR(WIRNode(node), PRs)
            os.append(o.node)
        os = WIRNode(list(flatten(os)))
        return os, PRs
    
    c = None

    if isinstance(v.node, (str, int, float, bool, type(None))): 
        if isinstance(v.node, str):
            # progarates if this is from function (attribute) or normal name
            return_name = v.node.replace('\n', '')
            if isAttribute:
                return_name = return_name + ':meth'
            v = WIRNode(return_name) # problematic later on (from_string in Agraph) # add variable tag!
        return v,  PRs # no id, implicid because of tree traversal
    
    p, PRs = GenPR(extract_from_node(v, 'operation'), PRs)
    I, PRs = GenPR(extract_from_node(v, 'input'), PRs)
    c, PRs = GenPR(extract_from_node(v, 'caller'), PRs)
    O, PRs = GenPR(extract_from_node(v, 'output'), PRs)
    
    if O.node is None:
        # this logic prevents loops - for some methods we cannot assign the id... (since we git objects)
        if is_empty_or_none_list(I.node) and c.node is None:# if we got only caller object and no other proveance, just return caller, do not add an id
            O = p
            return O, PRs
        elif isinstance(p.node, list):
            O = WIRNode([(op + add_id()) for op in p.node])
        else:
            O = WIRNode(p.node + add_id())
        
        
    input = I.node if isinstance(I.node, list) else [I.node]
    output  = O.node if isinstance(O.node, list) else [O.node]
    caller = c.node if isinstance(c.node, list) else [c.node]
    operation = p.node if isinstance(p.node, list) else [p.node]

    for _i, _c, _p, _o in itertools.product(input, caller, operation, output):
        PRs.append((_i, _c, _p, _o))
    return O, PRs

def remove_assignments(PRs_filtered: Set[PRType]) -> Set[PRType]:
    # get rid of assign nodes - move to another function
    assign_operations = [(pr_id,{'input': pr[0], 'output': pr[3]}) for pr_id, pr in enumerate(PRs_filtered) if pr[2] is not None and 'Assign' == remove_id(pr[2]) and pr[1] is None]  
    added_prs = []
    indices_to_delete = []
    
    # assign output_indices to the prs belonging to input nodes (where they are the output)
    for assign_pr_id, assign in assign_operations:
        revelant_input_prs = []
        for pr_id, pr2 in enumerate(PRs_filtered):
            if pr2[3] == assign['input']:
                revelant_input_prs.append(pr_id)
                indices_to_delete.append(pr_id)
            if pr2[0] == assign['input']:
                for rel_input_id in revelant_input_prs:
                    added_prs.append( (PRs_filtered[rel_input_id][0], PRs_filtered[rel_input_id][1], PRs_filtered[rel_input_id][2], pr2[3]))
                    
        indices_to_delete.append(assign_pr_id)
    PRs_filtered_new = [pr for i, pr in enumerate(PRs_filtered) if i not in indices_to_delete]
    PRs_filtered_new = PRs_filtered_new + added_prs
    
    return PRs_filtered_new
    

def GenWIR(root: ast.AST, output_filename: str, if_draw_graph: bool) -> Tuple[nx.DiGraph, Set[PRType], tuple[Set, Set, Set, Set]]:
    """
    Generates the WIR (Workflow Intermediate Representation) from an AST.
    
    :param root: Root node of the AST.
    :return: WIR graph G.
    """
    PRs = []
    
    for child in iter_child_nodes(root):
        _, PRs_prime = GenPR(WIRNode(child), PRs)
        PRs = merge_prs(PRs, PRs_prime)
        
    PRs = [pr for pr in PRs if pr[2] is not None]
    
    PRs_filtered = filter_PRs(PRs) # filter PRs (problematic operations)
    PRs_filtered = fix_bibartie_issue_import_from(PRs_filtered) # fix bipartie issue for import from statements
    
    bipartie_check = check_bipartie(PRs_filtered)
    
    PR_filtered_no_assign = remove_assignments(PRs_filtered)
    logger.warning(f"Graph is bipartie: {bipartie_check}")
                           
    with open(output_filename.replace('.png', '.txt'), 'w') as f:
        for pr in PR_filtered_no_assign:
            f.write(f"{pr}\n")
        
    G, (input_nodes, output_nodes, caller_nodes, operation_nodes) = construct_bipartite_graph(PR_filtered_no_assign, output_filename=output_filename, if_draw_graph = if_draw_graph)
    
    return G, PR_filtered_no_assign, (input_nodes, output_nodes, caller_nodes, operation_nodes)

def filter_PRs(PRs: Set[PRType]) -> Set[PRType]:
    """
    Optimizes the graph by getting rid of double operation nodes.
    Where the nodes are  Some Caller -> (Operator A + some id) -> (Output A + some other id).
    
    :param PRs: Set of PRs.
    :return: Filtered set of PRs.
    """
    filtered_PRs = []
    problematic_operations = dict()
    operations = set([ o for (_, _, o, _) in PRs])

    # get double operations
    for (I, c, p, O) in PRs:
        if c is not None and p in operations and O in operations and remove_id(p) == remove_id(O): 
            if O not in problematic_operations:
                problematic_operations[O] = c

    for (I, c, p, O) in PRs:
        if p in problematic_operations and c is None: # p is an operation produced by another PR
            original_caller = problematic_operations[p] # add original caller 
            filtered_PRs.append((I, original_caller, p, O))
            # remove p from problematic operations
        elif O in problematic_operations and c is not None:
            continue
        else:
            filtered_PRs.append((I,c,p,O))
    return filtered_PRs
        
def fix_bibartie_issue_import_from(PRs: Set[PRType]) -> Set[PRType]:
    """
    Fixes the bipartie issue for import from statements.
    
    :param PRs: Set of PRs.
    :return: Fixed set of PRs.
    """
    filtered_PRs = []
    imported = {}
    for (I, c, p, O) in PRs:
        if remove_id(p) == 'ImportFrom':
            imported[O] = O + add_id()
            filtered_PRs.append((I, c, p, imported[O]))
            filtered_PRs.append((None, imported[O], O, None))
        else:
            filtered_PRs.append((I, c, p, O))
    return filtered_PRs

def construct_bipartite_graph(PRs: Set[PRType], output_filename: str, if_draw_graph: bool) -> tuple[nx.DiGraph, tuple[Set, Set, Set, Set]]:
    """
    Constructs a bipartite graph from the given PRs.
    
    :param PRs: Set of PRs.
    :return: Bipartite graph G.
    """
    G = nx.DiGraph()
    input_nodes = set()
    operation_nodes = set()
    caller_nodes = set()
    output_nodes = set()
        
    for (I, c, p, O) in PRs:
        input_nodes.update([e for e in (I, ) if e is not None])
        operation_nodes.add(p)
        if c is not None:
            caller_nodes.add(c)
        if O is not None:
            output_nodes.add(O)
        
        for input_node in [e for e in (I, ) if e is not None]:
            G.add_edge(input_node, p, edge_type='input_to_operation', color='blue')
        if c is not None:
            G.add_edge(c, p, edge_type='caller_to_operation', color='red')
        if O is not None:
            G.add_edge(p, O, edge_type='operation_to_output', color='black')

    nx.set_node_attributes(G, {node: 'input' for node in input_nodes}, 'role')    
    nx.set_node_attributes(G, {node: 'caller' for node in caller_nodes}, 'role')   
    nx.set_node_attributes(G, {node: 'operation' for node in operation_nodes}, 'role')
    nx.set_node_attributes(G, {node: 'output' for node in output_nodes}, 'role')   
    
    all_nodes = input_nodes | output_nodes | caller_nodes | operation_nodes
    nx.set_node_attributes(G, {node: remove_id(node) for node in all_nodes}, 'label')   
    
    if if_draw_graph:
        draw_graph(G, input_nodes, output_nodes, caller_nodes, operation_nodes, output_filename)
    return G, (input_nodes, output_nodes, caller_nodes, operation_nodes)

def draw_graph(G, input_nodes, output_nodes, caller_nodes, operation_nodes, output_filename):
    
    labels = {node: remove_id(node) for node in G.nodes()}
    
    plt.figure(figsize=(200, 40))
    pos = graphviz_layout(G, prog='dot')

    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_shape='o') 
    nx.draw_networkx_nodes(G, pos, nodelist=caller_nodes, node_shape='o')
    nx.draw_networkx_nodes(G, pos, nodelist=operation_nodes, node_shape='s') 
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_shape='o')

    edges = G.edges(data=True)
    edge_colors = [d['color'] for (_, _, d) in edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    plt.legend()
    plt.savefig(output_filename)
    plt.close()
    

if __name__ == "__main__":
    name = 'titanic-advanced-feature-engineering-tutorial'
    file_path = f'data/titanic_mvp_wir/{name}/script.py'
    location_related_attributes = ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']

    with open(file_path, 'r') as file:
        file_content = file.read()
    file_lines = file_content.split('\n')
        
    parsed_ast = ast.parse(file_content)
    wir, _ = GenWIR(parsed_ast, output_filename=f'output/wir-{name}.png', if_draw_graph=True)
    print("Generated WIR:", wir)
