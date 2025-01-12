import json
import os
import requests
import time
import nbformat
import re
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import logging

from ApexDAG.scripts.crawl_notebooks.utils import InvalidNotebookException

class NotebookProcessor:
    # Class-level caches for regex patterns
    alias_pattern_cache = {}
    object_pattern_cache = {}
    attribute_pattern_cache = {}

    def __init__(self, save_dir, log_file):
        self.save_dir = save_dir
        self.logs_dir = os.path.join(save_dir, 'logs')

        self.logger = logging.getLogger(f'NotebookProcessor + ({log_file})')
        self.logger.setLevel(logging.INFO)

        os.makedirs(self.logs_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(os.path.join(self.logs_dir, log_file))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def load_filenames(self, json_file):
        """
        Load filenames from a JSON file.
        """
        try:
            with open(json_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"File not found: {json_file}")
            return []
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding JSON file: {e}")
            return []

    @classmethod
    def get_alias_pattern(cls, import_table):
        alias_key = tuple(import_table['alias'])
        if alias_key not in cls.alias_pattern_cache:
            if import_table['alias'].empty:
                alias_pattern = r'(?!)'  # This pattern matches nothing
            else:
                alias_pattern = '|'.join(re.escape(alias) for alias in import_table['alias'])
            cls.alias_pattern_cache[alias_key] = alias_pattern
        return cls.alias_pattern_cache[alias_key]

    @classmethod
    def get_object_pattern(cls, alias_pattern):
        if alias_pattern not in cls.object_pattern_cache:
            object_pattern = r'\b(?:' + alias_pattern + r')\([^)]+\)'  # Object calls (e.g., Universe())
            cls.object_pattern_cache[alias_pattern] = object_pattern
        return cls.object_pattern_cache[alias_pattern]

    @classmethod
    def get_attribute_pattern(cls, alias_pattern):
        if alias_pattern not in cls.attribute_pattern_cache:
            attribute_pattern = r'\b(?:' + alias_pattern + r')\.[a-zA-Z0-9_]+'
            cls.attribute_pattern_cache[alias_pattern] = attribute_pattern
        return cls.attribute_pattern_cache[alias_pattern]

    @staticmethod
    def get_import_table(notebook_content):
        """
        Extract import statements and build an import table DataFrame.
        """
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
                elif match.group(3):  # 'from module import ...'
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

    @staticmethod
    def simple_check_aliases_in_code_without_import(cell_code, import_table):
        """
        Check if each alias in the import_table is used in the provided code.
        """
        code_lines = cell_code.split('\n')

        def check_alias_in_line(alias):
            return any((alias in line.split('#')[0]) and not (line.startswith("import")) and (not line.startswith("from")) for line in code_lines) # omit line if starts with comments

        import_table['is_in_code'] = import_table['alias'].apply(check_alias_in_line)
        return import_table
    
    @classmethod
    def get_object_pattern(cls, alias_pattern):
        return r'({})\.\w+\('.format(alias_pattern)

    @classmethod
    def get_attribute_pattern(cls, alias_pattern):
        return r'({})\.\w+'.format(alias_pattern)

    @classmethod
    def methods_used_in_code(cls, code, import_table):
        """
        Extract method calls and update the import_table with a 'methods_used' column.
        """
        code = [line for line in code.splitlines() if not line.startswith("import") and not line.startswith("from")]
        code = [line.split('#')[0] for line in code if not line.strip().startswith("#")] # omit line if starts with comments
        code = "\n".join(code)

        alias_pattern = cls.get_alias_pattern(import_table)
        object_pattern = cls.get_object_pattern(alias_pattern)
        attribute_pattern = cls.get_attribute_pattern(alias_pattern)

        object_calls = list(re.finditer(object_pattern, code))
        attribute_accesses = list(re.finditer(attribute_pattern, code))

        methods_used = defaultdict(Counter)
        methods_placement = defaultdict(lambda: defaultdict(list))

        for match in object_calls:
            call = match.group()
            alias = call.split('(')[0]
            methods_used[alias][alias] += 1
            methods_placement[alias][alias].append((match.start(), match.end()))

        for match in attribute_accesses:
            call = match.group()
            alias = call.split('.')[0]
            method_name = call.split('.')[-1]
            methods_used[alias][method_name] += 1
            methods_placement[alias][method_name].append((match.start(), match.end()))

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
                "methods_used": dict(methods_used[alias]),
                "methods_placement": dict(methods_placement[alias])
            }

        return json_output

    def process_cells(self, cell_code):
        """
        Process cells to extract methods used.
        """
        import_table = self.get_import_table(cell_code)
        import_table = self.simple_check_aliases_in_code_without_import(cell_code, import_table)
        return self.methods_used_in_code(cell_code, import_table)

    
    def extract_code(self, notebook):
        if notebook is None:
            return None
        cells = notebook.get("cells", [])
        code_cells = [cell['source'] for cell in cells if (cell.get("cell_type") == "code") and (cell['source'] is not None)]
        return '\n'.join(code_cells)
    
    