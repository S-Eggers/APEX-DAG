import random


def add_id():
    return ':id' + str(random.randint(0, 2500))

def remove_id(node_name: str):
    return node_name.split(':')[0]

def is_empty_or_none_list(I):
    return I is None or (isinstance(I, list) and all(x is None for x in I))

def flatten(lst):
    """
    Recursively flattens a list while preserving strings.
    
    :param lst: List to be flattened.
    :return: Flattened list.
    """
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item
            

def get_relevant_code(node, file_lines):
    '''
    Utility, returns code pertaining to a node
    '''
    if hasattr(node, '__dict__'):
        if 'lineno' in vars(node) and 'col_offset' in vars(node) and 'end_col_offset' in vars(node):
            return file_lines[node.lineno-1][node.col_offset:node.end_col_offset]

# delete, only for inspection
def print_relevant_code(element):
    '''
    Utility, prints code pertaining to a node
    '''
    if isinstance(element, list):
        for el in element:
            print_relevant_code(el)
    elif isinstance(element, str):
        print(element)
    elif element is not None:
        get_relevant_code(element)
    else:
        print("None")           
            