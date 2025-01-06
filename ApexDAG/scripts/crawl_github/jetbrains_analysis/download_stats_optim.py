import json
import os
import requests
import time
import nbformat
import re
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import logging

# Precompiled regex patterns to optimize repeated calls
alias_pattern_cache = {}
object_pattern_cache = {}
attribute_pattern_cache = {}

def get_alias_pattern(import_table):
    alias_key = tuple(import_table['alias'])
    if alias_key not in alias_pattern_cache:
        alias_pattern = '|'.join(re.escape(alias) for alias in import_table['alias'])
        alias_pattern_cache[alias_key] = alias_pattern
    return alias_pattern_cache[alias_key]

def get_object_pattern(alias_pattern):
    if alias_pattern not in object_pattern_cache:
        object_pattern = r'\b(?:' + alias_pattern + r')\([^\)]*\)'  # Object calls (e.g., Universe())
        object_pattern_cache[alias_pattern] = object_pattern
    return object_pattern_cache[alias_pattern]

def get_attribute_pattern(alias_pattern):
    if alias_pattern not in attribute_pattern_cache:
        attribute_pattern = r'\b(?:' + alias_pattern + r')\.[a-zA-Z0-9_]+'
        attribute_pattern_cache[alias_pattern] = attribute_pattern
    return attribute_pattern_cache[alias_pattern]

def load_filenames(json_file):
    '''
    Download ntbs_list.json file from the S3 bucket
    '''
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"File not found: {json_file}")
        return []
    except json.JSONDecodeError as e:
        logging.warning(f"Error decoding JSON file: {e}")
        return []

def methods_used_in_code(code, import_table):
    """
    This function extracts method calls used in the provided code based on the import table
    and updates the import_table dataframe with an additional 'methods_used' column.
    """
    code = [line for line in code.splitlines() if not line.startswith("import") and not line.startswith("from")]
    code = "\n".join(code)
    
    alias_pattern = get_alias_pattern(import_table)
    object_pattern = get_object_pattern(alias_pattern)
    attribute_pattern = get_attribute_pattern(alias_pattern)

    object_calls = re.findall(object_pattern, code)
    attribute_accesses = re.findall(attribute_pattern, code)

    methods_used = defaultdict(Counter)
    
    # Process Object Calls
    for call in object_calls:
        alias = call.split('(')[0]
        methods_used[alias][alias] += 1

    # Process Attribute Access
    for call in attribute_accesses:
        alias = call.split('.')[0]
        method_name = call.split('.')[-1]
        methods_used[alias][method_name] += 1

    # Build the JSON object
    json_output = {module: {} for module in import_table['modules']}
    for _, row in import_table.iterrows():
        alias = row['alias']
        module = row['modules']
        submodule = row['submodules']
        methods = row['methods']
        simple_regex_check = row['is_in_code']

        json_output[module][alias] = {
            "module": module,
            "alias": alias,
            "submodule": submodule,
            "methods": methods,
            'regex_presence_check': simple_regex_check,
            "methods_used": dict(methods_used[alias]),  # Convert Counter to dict
        }

    return json_output

def get_import_table(notebook_content):
    lines = [line.strip() for line in notebook_content.split("\n")]
    import_statements = [line for line in lines if line.startswith("import") or line.startswith("from")]
    
    data = []
    import_pattern = re.compile(r"^(?:import\s+(\S+)(?:\s+as\s+(\S+))?|from\s+(\S+)\s+import\s+([\w,]+(?:\s*,\s*[\w,]+)*)(?:\s+as\s+([\w,]+))?)$")
    
    for statement in import_statements:
        module, alias, submodule, methods = "", "-", "", "-"
        match = import_pattern.search(statement)
        
        if match:
            if match.group(1):  # 'import module as alias'
                module = match.group(1)
                alias = match.group(2) if match.group(2) else module
            elif match.group(3):  # 'from module import ...' statement
                module = match.group(3)
                methods = match.group(4) if match.group(4) else "-"
                alias = match.group(5) if match.group(5) else methods

                if "," in methods:
                    methods = [method.strip() for method in methods.split(",")]
                    if "." in module:
                        module, submodule = module.split(".", 1)
                    for method in methods:
                        data.append([module, method, submodule, method])
                    continue

            if "." in module:
                module, submodule = module.split(".", 1)
                
        if alias != "-":
            data.append([module, alias, submodule, methods])
    df = pd.DataFrame(data, columns=["modules", "alias", "submodules", "methods"])
    df = df.drop_duplicates(subset=["modules", "alias", "submodules", "methods"])  
    return df

def simple_check_aliases_in_code_without_import(cell_code, import_table):
    """
    Checks if each alias in the import_table is used in the provided code.
    """
    code_lines = cell_code.split('\n')

    def check_alias_in_line(alias):
        return any(alias in line and 'import' not in line for line in code_lines)

    import_table['is_in_code'] = import_table['alias'].apply(check_alias_in_line)
    return import_table

def process_cells(cell_code):
    '''
    Only catches methods directly induced from a module! 
    Not counting methods from object classes.
    '''
    import_table = get_import_table(cell_code)
    import_table = simple_check_aliases_in_code_without_import(cell_code, import_table)
    methods_object = methods_used_in_code(cell_code, import_table)
    return methods_object


def get_notebook_cells(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        notebook_content = response.content.decode("utf-8")
        notebook = nbformat.reads(notebook_content, as_version=4)
        cells = notebook.get("cells", [])
        
        code_cells = [cell['source'] for cell in cells if cell.get("cell_type") == "code"]
        return '\n'.join(code_cells)
    except requests.exceptions.RequestException as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return []
    except Exception as e:
        logging.warning(f"Error processing notebook {url}: {e}")
        return []

def download_notebooks(json_file, bucket_url, save_dir, output_file_name='annotated_jetbrains_optim_50k.json', limit=None, delay=0.01):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filenames = load_filenames(json_file)
    if not filenames:
        logging.warning("No filenames found.")
        return

    if limit:
        filenames = filenames[:limit]

    all_annotations = {}

    for filename in tqdm(filenames, desc="Processing notebooks"):
        try:
            file_url = f"{bucket_url}{filename}"
            code = get_notebook_cells(file_url)
            annotations_object = process_cells(code)
            all_annotations[filename] = annotations_object
        except Exception as e:
            logging.warning(f"Error processing {filename}: {e}")
            continue

    output_file_path = os.path.join(save_dir, output_file_name)
    try:
        with open(output_file_path, 'w') as f:
            json.dump(all_annotations, f, indent=4)
        logging.warning(f"Annotations successfully saved to {output_file_path}")
    except Exception as e:
        logging.warning(f"Failed to save annotations: {e}")

if __name__ == "__main__":
    JSON_FILE = "data/ntbs_list.json"
    BUCKET_URL = "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"
    SAVE_DIR = "data/notebooks"
    LIMIT = 50000

    download_notebooks(JSON_FILE, BUCKET_URL, SAVE_DIR, limit=LIMIT)
