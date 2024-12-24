import ast
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from ApexDAG.vamsa.utils import add_id, remove_id, is_empty_or_none_list, flatten, get_relevant_code

from networkx.drawing.nx_agraph import graphviz_layout
from ast import iter_child_nodes


def extract_from_node(node, field): # main function, not defined in the paper
    """Extracts information from a node."""
    if node is None:
        return None
    
    match node.__class__.__name__:
        case "Assign":
            if field == "operation":
                return  node.__class__.__name__ + add_id()
            elif field == "input":
                return node.value 
            elif field == "output":
                return node.targets
        case "Call":
            if field == "operation":
                return node.func
            elif field == "input":
                return node.args
        case "Attribute":
            if field == "caller":
                return node.value
            elif field == "operation":
                return node.attr 
            elif field == "output":
                return node.attr + add_id() # also needs id attribute
        case "Name":
            if field == "output":
                return node.id # + add_id() # also needs id attribute
        case "Constant":
            if field == "output":
                return f"{node.value}{add_id()}" # also needs id attribute
        case 'Import':
            if field == "operation":
                return node.__class__.__name__ + add_id()
            elif field == "output":
                return node.names
        case 'Module':
            pass
        case 'alias':
            if field == "output":
                return node.asname if node.asname is not None else node.name # also needs id attribute
            elif field == "caller":
                return node.name if node.asname is not None else node.name + add_id() 
            if field == "operation":
                return node.__class__.__name__ + add_id()
        case 'ImportFrom':
            if field == "operation":
                return node.__class__.__name__ + add_id()
            elif field == "output":
                return node.names
            elif field == "caller":
                return node.module
        case 'Store':
            pass # TODO
        case 'Subscript':
            if field == "operation":
                return node.__class__.__name__  + add_id()
            elif field == "input":
                return node.slice
            elif field == "caller":
                return node.value
            elif field == "output":
                return node.__class__.__name__  + add_id()
        case 'Tuple': # this omits the modelling of tuple...
            if field == "output":
                return node.elts
        case 'Slice':
            if field == "operation":
                return node.__class__.__name__  + add_id()
            elif field == "input":
                return [value if value is not None else '' + add_id() for value in (node.lower, node.upper, node.step)]
        case 'List':
            if field == "input":
                return node.elts
            elif field == "operation":
                return node.__class__.__name__ + add_id() 
        case 'keyword':
            if field == "output":
                return node.arg
            elif field == "input":
                return node.value
        case 'Expr': # TODO
            if field == 'output':
                return node.value
    return None


def GenPR(v, PRs):
    """
    Processes a single AST node to generate WIR variables and update PRs.
    
    :param v: AST node.
    :param PRs: Set of PRs generated so far.
    :return: Tuple containing a set of WIR variables and updated PRs.
    """
    if v is None:
        return None, PRs
    
    if isinstance(v, list): # hanfles input/ output lists
        os = []
        for node in v:
            o, PRs = GenPR(node, PRs)
            os.append(o)
        os = list(flatten(os))
        return os, PRs
    
    c = None

    if isinstance(v, (str, int, float, bool, type(None))): 
        PRs.add((None, None, None, v))
        return v,  PRs # no id, implicid because of tree traversal
    
    p, PRs = GenPR(extract_from_node(v, 'operation'), PRs)
    I, PRs = GenPR(extract_from_node(v, 'input'), PRs)
    c, PRs = GenPR(extract_from_node(v, 'caller'), PRs)
    O, PRs = GenPR(extract_from_node(v, 'output'), PRs)
    
    if O is None:
        # this logic prevents loops - for some methods we cannot assign the id... (since we git objects)
        if is_empty_or_none_list(I) and c is None:# if we got only caller object and no other proveance, just return caller, do not add an id
            O = p
        elif isinstance(p, list):
            O = [op + add_id() for op in p]
        else:
            O = p + add_id()
        
        
    I = I if isinstance(I, list) else [I]
    O = O if isinstance(O, list) else [O]
    c = c if isinstance(c, list) else [c]
    p = p if isinstance(p, list) else [p]

    for i, caller, operation, o in itertools.product(I, c, p, O):
        PRs.add((i, caller, operation, o))

    return O, PRs

def GenWIR(root, output_filename='output/wir.png'):
    """
    Generates the WIR (Workflow Intermediate Representation) from an AST.
    
    :param root: Root node of the AST.
    :return: WIR graph G.
    """
    PRs = set()
    
    for child in iter_child_nodes(root):
        _, PRs_prime = GenPR(child, PRs)
        PRs = PRs.union(PRs_prime)
        
    PRs = {pr for pr in PRs if pr[2] is not None}
    
    for pr in PRs:
        print("Unfiltered pr: ", pr)
        
    PRs_filtered = filter_PRs(PRs)
    
    for pr in PRs:
        print("Filtered pr: ", pr)
        
    G = construct_bipartite_graph(PRs_filtered, output_filename=output_filename)
    
    return G

def filter_PRs(PRs):
    """
    Optimizes the graph by getting rid of double operation nodes.
    Where the nodes are  Some Caller -> (Operator A + some id) -> (Output A + some other id) and
    
    :param PRs: Set of PRs.
    :return: Filtered set of PRs.
    """
    filtered_PRs = set()
    problematic_operations = dict()
    operations = set([ o for (I, c, o, O) in PRs])
    
    for (I, c, p, O) in PRs:
        if c is not None and p in operations and O in operations and remove_id(p) == remove_id(O): 
            if O not in problematic_operations:
                problematic_operations[O] = c
            elif problematic_operations[O] != c: # may be a multiinput operation!
                raise ValueError("Multiple problematic operations caused same outputs: %s" % O)
    
    for (I, c, p, O) in PRs:
        # p is an output of a problematic operation node...
        if p in problematic_operations and c is None:
            # add original caller 
            original_caller = problematic_operations[p]
            filtered_PRs.add((I, original_caller, p, O))
        elif O in operations and p in operations: 
            if I is not None: 
                pass
                # later raise value error 
                #raise ValueError("Input should be None, but is %s" % I)
        else:
            filtered_PRs.add((I,c,p,O))
    return filtered_PRs
            

def construct_bipartite_graph(PRs, output_filename):
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
        if O is not None: # redundant
            output_nodes.add(O)
        
        for input_node in [e for e in (I, ) if e is not None]:
            G.add_edge(input_node, p, edge_type='input_to_operation', color='blue')
        if c is not None:
            G.add_edge(c, p, edge_type='caller_to_operation', color='red')
        if O is not None:
            G.add_edge(p, O, edge_type='operation_to_output', color='black')

    nx.set_node_attributes(G, {node: 0 for node in input_nodes}, 'bipartite')    
    nx.set_node_attributes(G, {node: 1 for node in caller_nodes}, 'bipartite')   
    nx.set_node_attributes(G, {node: 2 for node in operation_nodes}, 'bipartite')
    nx.set_node_attributes(G, {node: 3 for node in output_nodes}, 'bipartite')   
    
    labels = {node: remove_id(node) for node in G.nodes()}
    print("Nodes in the graph:", G.nodes())
    plt.figure(figsize=(20, 20))
    pos = graphviz_layout(G, prog='dot')

    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_shape='o') 
    nx.draw_networkx_nodes(G, pos, nodelist=caller_nodes, node_shape='o')
    nx.draw_networkx_nodes(G, pos, nodelist=operation_nodes, node_shape='s') 
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_shape='o')

    edges = G.edges(data=True)
    edge_colors = [d['color'] for (u, v, d) in edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    plt.legend()
    plt.savefig(output_filename)
    plt.close()
    
    return G

# Example usage
if __name__ == "__main__":
    file_path = 'data/raw/test_vamsa.py'
    location_related_attributes = ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']

    with open(file_path, 'r') as file:
        file_content = file.read()
    file_lines = file_content.split('\n')
        
    parsed_ast = ast.parse(file_content)
    wir = GenWIR(parsed_ast, output_filename='output/wir.png')
    print("Generated WIR:", wir)
