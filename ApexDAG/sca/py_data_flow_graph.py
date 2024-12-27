import re
import ast
import astpretty
import networkx as nx
from typing import Optional

from ApexDAG.util.draw import Draw
from ApexDAG.sca.ast_graph import ASTGraph
from ApexDAG.util.logging import setup_logging
from ApexDAG.sca.py_util import get_operator_description
from ApexDAG.sca.graph_utils import convert_multidigraph_to_digraph, get_subgraph
from ApexDAG.sca.constants import NODE_TYPES, EDGE_TYPES, VERBOSE


class PythonDataFlowGraph(ASTGraph, ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self._setup("module")
        
        self.imported_names = {}
        self.import_from_modules = {}
        self.classes = {}
        self.functions = {}
        self._state = {}
        self._logger = setup_logging("py_data_flow_graph", VERBOSE)
        self._debug_context = []
        self._branches = []
    
    def visit_Import(self, node: ast.Import):
        # Track imported modules
        for alias in node.names:
            self.imported_names[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Track imported modules from 'import from' statements
        module = node.module
        for alias in node.names:
            self.imported_names[alias.asname or alias.name] = module
            self.import_from_modules[alias.asname or alias.name] = module

    def visit_Assign(self, node: ast.Assign):
        target = node.targets[0]
        value = node.value       

        # Get the versioned name of the target variable using the line number
        target_names = self._get_names(target)
        if not target_names:
            self._logger.error(f"Could not get target names for {ast.get_source_segment(self.code, target)}")
            target_names = []
            
        for target_name in target_names:
            # Create a new version for this variable
            self._set_current_variable(self._get_versioned_name(target_name, node.lineno))
            self._set_current_target(target_name)
            
            if isinstance(value, ast.Lambda):
                self.functions[target.id] = {"node": self.current_variable}
            else:
                self._add_node(self.current_variable, NODE_TYPES["VARIABLE"])

            value.parent = node
            self.visit(value)
            
            # Update the latest version of this variable
            if target_name not in self.variable_versions:
                self.variable_versions[target_name] = []
                previous_version = None
            else:
                previous_version = self._get_last_variable_version(target_name)
            
            self.variable_versions[target_name].append(self.current_variable)
            
            if previous_version and not self._has_edge(previous_version, self.current_variable):
                self._add_edge(
                    previous_version, 
                    self.current_variable, 
                    "reassign", 
                    EDGE_TYPES["OMITTED"], 
                    node.lineno, 
                    node.col_offset, 
                    node.end_lineno, 
                    node.end_col_offset
                )

        self._set_current_variable(None)
        self._set_current_target(None)
        self._set_last_variable(None)
        self.edge_for_current_target = {}
        
    def visit_AugAssign(self, node: ast.AugAssign):
        # get operator
        try:
            operator =  node.op.__class__.__name__.lower()
        except AttributeError:
            self._logger.debug(f"Could not get operator for {ast.get_source_segment(self.code, node)}")
            operator = None

        target = node.target
        base_name = self._get_base_name(target)
        current_target = self._get_versioned_name(base_name, node.lineno)
        self._add_node(current_target, NODE_TYPES["VARIABLE"])
        self._add_edge(
            self._get_last_variable_version(base_name), 
            current_target, 
            operator, 
            EDGE_TYPES["CALLER"],
            node.lineno, 
            node.col_offset, 
            node.end_lineno, 
            node.end_col_offset
        )
                    
        node.targets = [target]      
        self.visit_Assign(node)
    
    def visit_Expr(self, node: ast.Expr):
        base_name = self._get_base_name(node.value)
        # just a name as an expression e.g. x, is not a data flow
        is_call = isinstance(node.value, ast.Call)
        is_first_order = is_call and isinstance(node.value.func, ast.Name)
        is_self_defined = base_name in self.functions
        is_imported = base_name in self.import_from_modules or base_name in self.imported_names
        
        if base_name and (not is_first_order or is_self_defined or is_imported):
            self._set_current_variable(self._get_versioned_name(base_name, node.lineno))
            self._set_current_target(base_name)
            self._add_node(self.current_variable, NODE_TYPES["VARIABLE"])
            
            node.value.parent = node
            self.visit(node.value)
            
            # Update the latest version of this variable
            if base_name not in self.variable_versions:
                self.variable_versions[base_name] = []
            
            self.variable_versions[base_name].append(self.current_variable)
            
        self._set_current_variable(None)
        self._set_current_target(None)
        self._set_last_variable(None)
        self.edge_for_current_target = {}
        
    def visit_Lambda(self, node: ast.Lambda):
        if hasattr(node, "parent") and isinstance(node.parent, ast.Assign):            
            name = self._get_names(node)[0]
            name = self._get_versioned_name(name, node.lineno)
            parent_name = self._get_names(node.parent.targets[0])[0]            
            context_name = f"{parent_name}_{name}"            
            self.functions[parent_name]["context"] = context_name            
            self.functions[parent_name]["args"] = self._process_arguments(node.args)
            self.functions[parent_name]["is_recursive"] = False
                
            self._store_state(self.context)
            self._setup(context_name, self.context)
            
            for arg in self.functions[parent_name]["args"]["args"]:
                argument_node = f"{arg}_{node.lineno}"
                self._add_node(argument_node, NODE_TYPES["VARIABLE"])
                self.variable_versions[arg] = [argument_node]            

            self.visit(node.body)
            self._store_state(context_name)
            self._restore_state(self.parent_context)
            
        else:
            self._logger.debug(f"Ignoring lambda function {ast.get_source_segment(self.code, node)} as it is assigned to a variable")
            self._logger.debug(ast.dump(node))
            # ToDo: Implement anonymous lambda functions
            super().generic_visit(node)
            arguments = self._process_arguments(node.args)
            for arg in arguments["args"]:
                argument_node = f"{arg}_{node.lineno}"
                self._add_node(argument_node, NODE_TYPES["VARIABLE"])
                self.variable_versions[arg] = [argument_node]            
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        function_name = self._get_names(node)[0]
        context_name = self._get_versioned_name(function_name, node.lineno)    
        self.functions[function_name] = {"node": context_name, "context": context_name, "is_recursive": self._check_resursion(node)}
        self.functions[function_name]["args"] = self._process_arguments(node.args)
        
        parent_context = self.context
        self._store_state(self.context)
        self._setup(context_name, self.context)

        for arg in self.functions[function_name]["args"]["args"]:
            argument_node = f"{arg}_{node.lineno}"
            self._add_node(argument_node, NODE_TYPES["VARIABLE"])
            self.variable_versions[arg] = [argument_node]
        
        for stmt in node.body:
            stmt.parent = node
            self.visit(stmt)

        self._store_state(context_name)
        self._restore_state(parent_context)
 
    def visit_Call(self, node: ast.Call):
        # we are calling a second order function, e.g. dataframe.dropna()
        if isinstance(node.func, ast.Attribute):         
            caller_object_name = self._get_caller_object(node.func.value)
            function_name = node.func.attr
            # we are calling a chained method, e.g. dataframe.dropna().dropna()
            # if isinstance(node.func.value, (ast.Call, ast.Attribute, ast.Name)):
            if hasattr(node.func, "value"):
                node.func.value.parent = node.func
                self.visit(node.func.value)

        # we are calling a function from the same module, e.g. dropna()
        # or a function from a module that was imported, e.g. read_csv()
        elif isinstance(node.func, ast.Name):
            caller_object_name = self._get_caller_object(node.func)
            function_name = node.func.id
        
        elif isinstance(node.func, (ast.Call, ast.Subscript)):
            caller_object_name = self._get_caller_object(node.func)
            function_name = "__call__"
            
        # we dont support other types of function calls like lambda functions yet
        else:
            raise NotImplementedError(f"Unsupported function call {ast.get_source_segment(self.code, node)} with node {ast.dump(node)}")
        
        # we are calling a method of an imported module, e.g. pd.read_csv() or an 
        # imported first order method, e.g. read_csv()
        if caller_object_name in self.imported_names or caller_object_name in self.import_from_modules: 
            self._process_library_call(node, caller_object_name, function_name)
        # we are calling a user defined class, e.g. dataframe = DataFrame(), ToDo: currently only working for Constructors
        elif caller_object_name in self.classes or function_name in self.classes:
            self._process_class_call(node, caller_object_name, function_name)
        # we are calling a user defined function, e.g. a lambda function stored in a variable
        elif function_name in self.functions:
            self._process_function_call(node, function_name)
        # we are calling a method of a object, e.g. dataframe = dataframe.dropna()
        elif self.current_target:
            self._process_method_call(node, caller_object_name, function_name)
        else:
            self._logger.debug(f"Ignoring function call {ast.get_source_segment(self.code, node)} as it contains no data flow")
            self._logger.debug(ast.dump(node))
            self._logger.debug(f"Caller object name {caller_object_name} not in {self.current_target}")

    def visit_BinOp(self, node: ast.BinOp):
        left_var, right_var = self._get_lr_values(node.left, node.right)
        
        # Add parent node
        node.left.parent = node
        node.right.parent = node
        
        # Visit left and right operands
        self.visit(node.left)
        self.visit(node.right)
        
        # get operator
        try:
            operator =  node.op.__class__.__name__.lower()
        except AttributeError:
            self._logger.debug(f"Could not get operator for {ast.get_source_segment(self.code, node)}")
            operator = None
        
        # Add edges from operands to the new variable version
        if self.current_variable and operator:        
            if left_var:
                left_version = self._get_last_variable_version(left_var)
                self._add_edge(
                    left_version, 
                    self.current_variable, 
                    operator, 
                    EDGE_TYPES["CALLER"],
                    node.lineno, 
                    node.col_offset, 
                    node.end_lineno, 
                    node.end_col_offset
                )
            if right_var:
                right_version = self._get_last_variable_version(right_var)
                self._add_edge(
                    right_version, 
                    self.current_variable, 
                    operator, 
                    EDGE_TYPES["CALLER"],
                    node.lineno, 
                    node.col_offset, 
                    node.end_lineno, 
                    node.end_col_offset
                )
                
    def visit_Compare(self, node: ast.Compare):
        # ToDo: not correctly implemented yet
        left_var, right_var = self._get_lr_values(node.left, node.comparators[0])
        
        node.left.parent = node
        node.comparators[0].parent = node
        
        self.visit(node.left)
        self.visit(node.comparators[0])
        
        # get operator
        operator = get_operator_description(node)
        if not operator:
            self._logger.debug(f"Could not get operator for {ast.get_source_segment(self.code, node)}")
            
        if self.current_variable and operator:
            if left_var:
                left_version = self._get_last_variable_version(left_var)
                self._add_edge(
                    left_version, 
                    self.current_variable, 
                    operator, 
                    EDGE_TYPES["CALLER"],
                    node.lineno, 
                    node.col_offset, 
                    node.end_lineno, 
                    node.end_col_offset
                )
            if right_var:
                right_version = self._get_last_variable_version(right_var)
                self._add_edge(
                    right_version, 
                    self.current_variable, 
                    operator, 
                    EDGE_TYPES["CALLER"],
                    node.lineno, 
                    node.col_offset, 
                    node.end_lineno, 
                    node.end_col_offset
                )
    
    def visit_Name(self, node: ast.Name):        
        if not isinstance(node.ctx, ast.Load):
            return

        var_name = node.id
        if var_name in self.imported_names or var_name in self.import_from_modules:
            self._logger.debug(f"Processing library attribute {var_name}")
            self._process_library_attr(node, var_name)
        else:
            var_version = self._get_last_variable_version(var_name)
            edge_type = EDGE_TYPES["INPUT"]
            # Determine the full code context for more complex structures            
            if hasattr(node, "parent") and isinstance(node.parent, ast.Subscript):
                code_segment, edge_type = self._process_subscript(node)
            else:
                code_segment = var_name
                
            code_segment = self._tokenize_method(code_segment)

            if self.current_variable:
                self._add_edge(
                    var_version, 
                    self.current_variable, 
                    code_segment, 
                    edge_type,
                    node.lineno, 
                    node.col_offset, 
                    node.end_lineno, 
                    node.end_col_offset
                )
                self._set_last_variable(var_version)
            
    def visit_Attribute(self, node: ast.Attribute):
        if hasattr(node, "value"):
            node.value.parent = node
            self.visit(node.value)
            
        var_name = None
        if self.last_variable:
            var_name = self.last_variable
        
        self._add_edge(
            var_name, 
            self.current_variable, 
            self._tokenize_method(node.attr), 
            EDGE_TYPES["CALLER"],
            node.lineno, 
            node.col_offset, 
            node.end_lineno, 
            node.end_col_offset
        )
        
    def visit_IfExp(self, node: ast.IfExp):        
        self._add_edge(
            self._get_last_variable_version(self.current_target), 
            self.current_variable, 
            "if", 
            EDGE_TYPES["CALLER"],
            node.lineno, 
            node.col_offset, 
            node.end_lineno, 
            node.end_col_offset
        )
        # visit if expression
        # ToDo: 'Name' object has no attribute 'parent'? 
        self.visit(node.body)
        # visit else expression
        self.visit(node.orelse)
    
    def visit_If(self, node: ast.If):
        if_branch = True
        # first check if we are in an if or else if statement
        if node.parent and isinstance(node.parent, ast.If) and isinstance(node.parent.orelse[0], ast.If):
            # we are actually in an else if statement
            parent_context = node.parent.parent_context
            if_context = f"else_if_{node.lineno}"
            self._visit_if_body(node.body, if_context, parent_context)
            self._branches.append((if_context, "else if", EDGE_TYPES["BRANCH"]))
            if_branch = False
        # we are not visiting an elif statement (if or else body)
        else:
            node_name = self._get_names(node)[0]
            var_version = self._get_versioned_name(node_name, node.lineno)
            previous_target = self.current_target
            self.current_target = var_version
            parent_context = self.context
            self._store_state(parent_context)
            
            if_context = f"{var_version}_if"
            self._visit_if_body(node.body, if_context, parent_context)
            self._branches.append((if_context, "if", EDGE_TYPES["BRANCH"]))
            
        # visit elif statements
        if node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node.parent_context = self.parent_context
            node.orelse[0].parent = node
            self.visit(node.orelse[0])
        # visit else statement
        elif len(node.orelse) > 0:
            else_context = f"else_{node.lineno}"
            self._visit_if_body(node.orelse, else_context, parent_context)
            self._branches.append((else_context, "else", EDGE_TYPES["BRANCH"]))
        # merge state
        if if_branch:
            self._merge_state(parent_context, *self._branches)
            self._branches = []
            self.current_target = previous_target
    
    def visit_While(self, node: ast.While):
        node_name = self._get_names(node)[0]
        var_version = self._get_versioned_name(node_name, node.lineno)
        
        parent_context = self.context
        self._store_state(parent_context)
        while_context = f"{var_version}_while"
        self._setup(while_context, parent_context)
        for stmt in node.body:
            stmt.parent = node
            self.visit(stmt)
        self._store_state(while_context)
        contexts = [(while_context, "loop", EDGE_TYPES["LOOP"])]
        
        if node.orelse and len(node.orelse) > 0:
            else_context = f"{var_version}_else"
            self._setup(else_context, parent_context)
            for stmt in node.orelse:
                stmt.parent = node
                self.visit(stmt)
            self._store_state(else_context)
            contexts.append((else_context, "else", EDGE_TYPES["BRANCH"]))
        
        self._merge_state(parent_context, *contexts)

    def visit_For(self, node: ast.For):
        node_name = self._get_names(node)[0]
        var_version = self._get_versioned_name(node_name, node.lineno)
        
        parent_context = self.context
        self._store_state(parent_context)
        for_context = f"{var_version}_for"
        self._setup(for_context, parent_context)
        for stmt in node.body:
            stmt.parent = node
            self.visit(stmt)
        self._store_state(for_context)
        contexts = [(for_context, "loop", EDGE_TYPES["LOOP"])]
        
        if node.orelse and len(node.orelse) > 0:
            else_context = f"{var_version}_else"
            self._setup(else_context, parent_context)
            for stmt in node.orelse:
                stmt.parent = node
                self.visit(stmt)
            self._store_state(else_context)
            contexts.append((else_context, "else", EDGE_TYPES["BRANCH"]))
            
        self._merge_state(parent_context, *contexts)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        name = node.name
        self.classes[name] = [name]      
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self.classes[name].append(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.classes[name].append(target.id)
                        
                    elif isinstance(target, ast.Attribute):
                        self.classes[name].append(target.attr)
    
    def visit_Delete(self, node: ast.Delete):
        pass 
    
    def visit_Return(self, node: ast.Return):
        pass
    
    def visit_Dict(self, node: ast.Dict):
        pass
    
    def visit_List(self, node: ast.List):
        pass
    
    def visit_Tuple(self, node: ast.Tuple):
        pass
    
    def visit_Set(self, node: ast.Set):
        pass

    def generic_visit(self, node: ast.AST):
        for child in ast.iter_child_nodes(node):
            child.parent = node

        super().generic_visit(node)
    
    def optimize(self):
        edges_to_remove = []
        nodes_to_remove = []
        edges_to_add = []

        for node_x in self._node_iterator():
            node_degrees = self._node_degree(node_x)
            if node_degrees["in"] == 0 and node_degrees["out"] == 0:
                self._logger.debug(f"Removing node {node_x} as it has no edges.")
                nodes_to_remove.append(node_x)

            node_type_x = self._get_node(node_x).get("node_type")
            # reconnect and remove end_if nodes
            if node_type_x == NODE_TYPES["IF"]:
                optimize = True
                new_edges = []        
                for next_node in self._successor_node_iterator(node_x):
                    for prev_node in self._predecessor_node_iterator(node_x):                       
                        for _, attributes in self._get_edge_iterator(node_x, next_node):
                            if attributes["edge_type"] in [EDGE_TYPES["LOOP"], EDGE_TYPES["BRANCH"]]:
                                optimize = False
                            new_edges.append((prev_node, next_node, attributes["code"], attributes["edge_type"]))
                # we need the intermediate node for branches that follow directly after that node
                if optimize:
                    nodes_to_remove.append(node_x)
                    edges_to_add += new_edges
                # continue to the next node as we already did the optimization for this node and its edges
                continue

            for node_y in self._adjacent_node_iterator(node_x):
                edges_between_nodes = list(self._get_edge_iterator(node_x, node_y))
                
                if len(edges_between_nodes) > 1:
                    for key, edge_data in edges_between_nodes:
                        if (edge_data["code"] in node_x.lower() or edge_data["code"].replace(" ", "_") in node_x.lower()) and edge_data["edge_type"] == EDGE_TYPES["INPUT"]:
                            self._logger.debug(f"Removing edge {node_x} -> {node_y} with code {edge_data['code']} as it is redundant.")
                            edges_to_remove.append((node_x, node_y, key))
        # remove nodes and edges
        for node in nodes_to_remove:
            self._remove_node(node)
        
        for node_x, node_y, key in edges_to_remove:
            self._remove_edge(node_x, node_y, key)
        
        # add new edges
        for node_x, node_y, code, edge_type in edges_to_add:           
            self._add_edge(node_x, node_y, code, edge_type)

    #-----------------------------------------------------------------------------------------------------------------------------------#
    #                                        Draw functions                                                                             #
    #                                        ToDo: speedup process of drawing graph image for live demo                                 #
    #-----------------------------------------------------------------------------------------------------------------------------------#
    def draw_all_subgraphs(self):
        for variable in self.variable_versions:
            self.draw(variable, variable)
        # draw the full graph
        self.draw()

    def draw(self, save_path: str=None, start_node: str=None):
        draw = Draw(NODE_TYPES, EDGE_TYPES)
        
        if start_node:
            G_copy = self._copy_graph()
            self._G = self._set_graph(get_subgraph(self._G, self.variable_versions, start_node))
            G = convert_multidigraph_to_digraph(self._G, NODE_TYPES)
            draw.dfg(G, save_path)
            self._G = self._set_graph(G_copy)
        else:
            G = convert_multidigraph_to_digraph(self._G, NODE_TYPES)
            draw.dfg(G, save_path)
        
    #-----------------------------------------------------------------------------------------------------------------------------------#
    #                                           abstract syntax tree helper functions                                                   #
    #-----------------------------------------------------------------------------------------------------------------------------------#
    def _check_resursion(self, node: ast.FunctionDef):
        function_name = node.name
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                if sub_node.func.id == function_name:
                    return True
        return False
            
    def _process_arguments(self, node: ast.arguments):
        # posonlyargs, kwonlyargs, vararg, kwarg probably not needed
        
        arguments = {"args": [], "defaults": []}
        for arg in node.args:
            arguments["args"].append(arg.arg)
        
        for default in node.defaults:
            arguments["defaults"].append(default)
            
        return arguments    
        
    def _visit_if_body(self, body: list[ast.AST], context: str, parent_context: str):
        self._setup(context, parent_context)
        for stmt in body:
            stmt.parent = context
            self.visit(stmt)
        self._store_state(context)
            
    def _process_method_call(self, node: ast.Call, caller_object_name: str, tokens: Optional[str]):
        previous_version = self._get_last_variable_version(caller_object_name)
        previous_version = previous_version if previous_version else self.last_variable
        tokens = self._tokenize_method(tokens)
        self._add_edge(
            previous_version, 
            self.current_variable, 
            tokens, 
            EDGE_TYPES["CALLER"],
            node.lineno, 
            node.col_offset, 
            node.end_lineno, 
            node.end_col_offset
        )

        for arg in node.args:
            if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                arg_name = self._get_names(arg)
                arg_name = arg_name[0] if arg_name else None              
                
                if arg_name:
                    arg_version = self._get_last_variable_version(arg_name)
                    code_segment = self._tokenize_method(arg_name)
                    self._add_edge(
                        arg_version, 
                        self.current_variable, 
                        code_segment, 
                        EDGE_TYPES["INPUT"],
                        node.lineno, 
                        node.col_offset, 
                        node.end_lineno, 
                        node.end_col_offset
                    )
        
    def _process_library_call(self, node: ast.Call, caller_object_name: str, tokens: str=None):
        # Add the import node and connect it                    
        import_node = self.imported_names[caller_object_name]
        self._add_node(import_node, NODE_TYPES["IMPORT"])
        tokens = tokens if tokens else ast.get_source_segment(self.code, node.func)
        tokens = self._tokenize_method(tokens)
        self._add_edge(
            import_node, 
            self.current_variable, 
            tokens, 
            EDGE_TYPES["CALLER"],
            node.lineno, 
            node.col_offset, 
            node.end_lineno, 
            node.end_col_offset
        )
        
        for arg in node.args:
            if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                arg_name = self._get_names(arg)
                arg_name = arg_name[0] if arg_name else None

                if arg_name:
                    arg_version = self._get_last_variable_version(arg_name)
                    code_segment = self._tokenize_method(arg_name)
                    self._add_edge(
                        arg_version, 
                        self.current_variable, 
                        code_segment, 
                        EDGE_TYPES["INPUT"],
                        node.lineno, 
                        node.col_offset, 
                        node.end_lineno, 
                        node.end_col_offset
                    )
        
        self._set_last_variable(import_node)
        
    def _process_class_call(self, node: ast.Call, caller_object_name: str, tokens: str=None):
        # Add the import node and connect it                    
        class_node = self.classes[tokens][0]
        self._add_node(class_node, NODE_TYPES["CLASS"])
        tokens = tokens if tokens else ast.get_source_segment(self.code, node.func)
        tokens = self._tokenize_method(tokens)
        self._add_edge(
            class_node, 
            self.current_variable, 
            tokens, 
            EDGE_TYPES["CALLER"],
            node.lineno, 
            node.col_offset, 
            node.end_lineno, 
            node.end_col_offset
        )
        
        for arg in node.args:
            if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                arg_name = self._get_names(arg)
                arg_name = arg_name[0] if arg_name else None

                if arg_name:
                    arg_version = self._get_last_variable_version(arg_name)
                    code_segment = self._tokenize_method(arg_name)
                    self._add_edge(
                        arg_version, 
                        self.current_variable, 
                        code_segment, 
                        EDGE_TYPES["INPUT"],
                        node.lineno, 
                        node.col_offset, 
                        node.end_lineno, 
                        node.end_col_offset
                    )
        
        self._set_last_variable(class_node)
        
    def _process_function_call(self, node: ast.Call, tokens: str=None):
        function_name = tokens
        function_context = self.functions[function_name]["context"]
        function_name_tokens = self._tokenize_method(function_name)

        mapping = dict()
        for index, arg in enumerate(node.args):
            mapping[self._get_base_name(arg)] = index
                    
        if not self.functions[function_name]["is_recursive"]:
            mapping = {k: self.functions[function_name]["args"]["args"][v] for k, v in mapping.items()}
            
            for keyword in node.keywords:
                mapping[self._get_base_name(keyword.value)] = keyword.arg
            
            inverse_mapping = {v: k for k, v in mapping.items()}
            current_context = self.context
            self._store_state(current_context)
            self._restore_state(function_context)
            
            for key, value in inverse_mapping.items():
                if key in self.variable_versions:
                    self.variable_versions[value] = self.variable_versions[key]
                    del self.variable_versions[key]
                        
            self._merge_state(current_context, (function_context, function_name_tokens, EDGE_TYPES["FUNCTION_CALL"]))
                    
    def _process_library_attr(self, node: ast.Attribute, caller_object_name: str):
        # Add the import node and connect it
        import_node = self.imported_names[caller_object_name]
        self._add_node(import_node, NODE_TYPES["IMPORT"])        
        tokens = self._tokenize_method(ast.get_source_segment(self.code, node))
        self._add_edge(
            import_node, 
            self.current_variable, 
            tokens, 
            EDGE_TYPES["CALLER"],
            node.lineno, 
            node.col_offset, 
            node.end_lineno, 
            node.end_col_offset
        )
        self._set_last_variable(import_node)
 
    def _process_subscript(self, node: ast.AST) -> tuple[str, int]:
        if isinstance(node.parent.slice, ast.Compare):
            code_segment = "filter"
            edge_type = EDGE_TYPES["CALLER"]
        elif isinstance(node.parent.slice, ast.Index):
            code_segment = "index"
            edge_type = EDGE_TYPES["CALLER"]
        elif isinstance(node.parent.slice, ast.Slice):
            code_segment = "slice"
            edge_type = EDGE_TYPES["CALLER"]
        elif isinstance(node.parent.slice, (ast.List, ast.Constant)):
            code_segment = "select"
            edge_type = EDGE_TYPES["CALLER"]
        else:
            self._logger.debug(f"Could not process subscript {ast.get_source_segment(self.code, node)}")
            self._logger.debug(ast.dump(node))
            code_segment = ast.get_source_segment(self.code, node.parent)
            edge_type = EDGE_TYPES["INPUT"]
        
        return code_segment, edge_type
    
    def _get_caller_object(self, value: ast.AST) -> str:
        names = self._get_names(value)

        if names and (
            names[0] in self.imported_names or 
            names[0] in self.import_from_modules or 
            names[0] in self.variable_versions
        ):
            return names[0]
        
        return self.current_target 
    
    def _get_versioned_name(self, var_name: str, lineno: int) -> str:
        return f"{var_name}_{lineno}"
    
    def _get_base_name(self, node: ast.AST):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            return self._get_base_name(node.value)
        elif isinstance(node, ast.Call):
            return self._get_base_name(node.func)

        return None

    def _get_names(self, node: ast.AST) -> Optional[list[str]]:            
        match node:
            case ast.Name():
                return [node.id]
            case ast.Attribute():
                name = self._get_names(node.value)
                return [name[0], node.attr] if name else [node.attr]
            case (ast.Tuple() | ast.List() | ast.Set()):
                names = [self._get_names(elt) for elt in node.elts if self._get_names(elt)]
                return [name[0] for name in names if name]
            case ast.FunctionDef():
                return [node.name]
            case ast.Subscript():
                return self._get_names(node.value)
            case (ast.If() | ast.IfExp()):
                return ["If"]
            case (ast.While() | ast.For()):
                return ["Loop"]
            case ast.Lambda():
                return ["Lambda"]
            case _:
                self._logger.debug(f"Could not get names for {ast.get_source_segment(self.code, node)}")
                self._logger.debug(ast.dump(node))
                return None
    
    def _tokenize_method(self, method: str) -> str:  
        split_by_period = method.split(".")
        
        # remove library alias as we dont want to learn from it
        for library_alias in self.imported_names.keys():
            if library_alias in split_by_period and library_alias not in self.import_from_modules:
                split_by_period.remove(library_alias)
        
        # split by underscore 
        split_by_underscore = []
        for item in split_by_period:
            split_by_underscore += item.split("_")
            
        split_by_camelcase = []
        for item in split_by_underscore:
            split_by_camelcase += re.sub('([a-z])([A-Z])', r'\1 \2', item).split()
        
        return " ".join(split_by_camelcase).lower()
    
    def _get_last_variable_version(self, variable: str, max_depth: int = 99) -> Optional[str]:
        if variable in self.variable_versions:
            return self.variable_versions.get(variable)[-1]

        elif self.parent_context and max_depth > 0:
            self._store_state(self.context)
            current_context = self.context
            self._restore_parent_state()
            variable_version = self._get_last_variable_version(variable, max_depth=max_depth - 1)
            self._restore_state(current_context)
            
            return variable_version
        else:
            return None
    
    def _get_lr_values(self, left: ast.AST, right: ast.AST) -> tuple[Optional[str], Optional[str]]:
        left_var = self._get_names(left)
        left_var = left_var[0] if left_var else None
        right_var = self._get_names(right)
        right_var = right_var[0] if right_var else None
        
        return left_var, right_var
    
    #-----------------------------------------------------------------------------------------------------------------------------------#
    #                                        state management functions                                                                 #
    #-----------------------------------------------------------------------------------------------------------------------------------#    
    def _set_current_variable(self, value: str):
        self.current_variable = value
        
    def _set_current_target(self, value: str):
        self.current_target = value
        
    def _set_last_variable(self, value: str):
        self.last_variable = value
        
    def _setup(self, context: str = None, parent_context: str = None):
        self.variable_versions = {}
        # variable name + variable version/line number
        self.current_variable: str = None
        # variable name
        self.current_target: str = None
        self.edge_for_current_target = {}
        self.payload = None
        self.last_variable: str = None
        self.context: str = context
        self.parent_context: str = parent_context
        self._G = nx.MultiDiGraph()
        
    def _restore_state(self, context: str):
        if context in self._state:            
            previous_state = self._state[context]
            self.variable_versions = previous_state["variable_versions"]
            self.current_variable = previous_state["current_variable"]
            self.current_target = previous_state["current_target"]
            self.edge_for_current_target = previous_state["edge_for_current_target"]
            self.payload = previous_state["payload"]
            self.last_variable = previous_state["last_variable"]
            self.context = previous_state["context"]
            self.parent_context = previous_state["parent_context"]
            self._G = previous_state["_G"]
        else:
            self._logger.debug(f"No state {context} to restore")
    
    def _restore_parent_state(self):
        self._restore_state(self.parent_context)
    
    def _merge_state(self, base_context: str, *args: tuple[str]):
        if base_context not in self._state:
            self._logger.error(f"No state found for base context {base_context}")
            return
        
        base_state = self._state[base_context]
        self._G = base_state["_G"]        
        new_variable_versions = {key: value[:] for key, value in base_state["variable_versions"].items()}
        branched_variables = {}
        looped_variables = {}
        
        for context, var_edge, edge_type in args:
            if context not in self._state:
                self._logger.warning(f"No state found for context {context}")
                continue
            
            is_loop = edge_type == EDGE_TYPES["LOOP"]            
            state = self._state[context]
            for key, value in state["variable_versions"].items():                
                if key in base_state["variable_versions"]:                    
                    last_var = base_state["variable_versions"][key][-1]
                    new_var = value[0]
                    self._add_edge(last_var, new_var, var_edge, edge_type)
                    new_variable_versions[key] += value
                else:
                    new_variable_versions[key] = value

                if key in branched_variables:
                    branched_variables[key].append(value[-1])
                else:
                    branched_variables[key] = [value[-1]]
                
                # ToDo: es kann sein, dass key hier nicht in base_state["variable_versions"] ist?
                if is_loop and key in base_state["variable_versions"]:
                    last_var = base_state["variable_versions"][key][-1]
                    looped_variables[key] = (last_var, value[-1])

            self._G = nx.compose(self._G, state["_G"])
            del self._state[context]
            
        for variable, loop in looped_variables.items():
            var_name = f"loop_{'_'.join(loop)}"
            self._add_node(var_name, NODE_TYPES["LOOP"])
            new_variable_versions[variable].append(var_name)
            
            self._add_edge(loop[1], var_name, "end_loop", EDGE_TYPES["LOOP"])
            self._add_edge(var_name, loop[0], "restart_loop", EDGE_TYPES["LOOP"])

        for variable, branches in branched_variables.items():
            if len(branches) > 1:
                var_name = f"branch_{'_'.join(branches)}"
                self._add_node(var_name, NODE_TYPES["IF"])
                new_variable_versions[variable].append(var_name)
                for node in branches:
                    self._add_edge(node, var_name, "end_if", EDGE_TYPES["BRANCH"])

        self.variable_versions = new_variable_versions
        self.current_variable = base_state["current_variable"]
        self.current_target = base_state["current_target"]
        self.edge_for_current_target = base_state["edge_for_current_target"]
        self.payload = base_state["payload"]
        self.last_variable = base_state["last_variable"]
        self.context = base_state["context"]
        self.parent_context = base_state["parent_context"]
            
    def _get_current_state(self):
        return {
            "variable_versions": self.variable_versions,
            "current_variable": self.current_variable,
            "current_target": self.current_target,
            "edge_for_current_target": self.edge_for_current_target,
            "payload": self.payload,
            "last_variable": self.last_variable,
            "context": self.context,
            "parent_context": self.parent_context,
            "_G": self._G,
        }
        
    def _store_state(self, context: str = None):
        current_state = self._get_current_state()
        self._state[context] = current_state
    
    #-----------------------------------------------------------------------------------------------------------------------------------#
    #                                            wrap networkx graph functions                                                          #
    #                                            possible extension: free choice of graph backend                                       #
    #-----------------------------------------------------------------------------------------------------------------------------------#
    def _copy_graph(self):
        return self._G.copy()
    
    def _set_graph(self, G):
        self._G = G

    def _get_node(self, node_identifier: str):
        return self._G.nodes[node_identifier]
    
    def _remove_node(self, node_identifier: str):
        self._G.remove_node(node_identifier)
    
    def _node_iterator(self):
        for node in self._G.nodes:
            yield node
            
    def _adjacent_node_iterator(self, node_identifier: str):
        for node in self._G[node_identifier]:
            yield node
    
    def _add_node(
        self, 
        node: str, 
        node_type: int
     ):
        if node not in self._G:
            self._G.add_node(node, label=node, node_type=node_type)
        else:
            self._logger.debug(f"Node {node} already exists in the graph")
            
    def _node_degree(self, node_identifier: str):
        return {
            "in": self._G.out_degree(node_identifier),
            "out": self._G.in_degree(node_identifier),
        }
        
    def _predecessor_node_iterator(self, node_identifier: str):
        for node in self._G.predecessors(node_identifier):
            yield node

    def _successor_node_iterator(self, node_identifier: str):
        for node in self._G.successors(node_identifier):
            yield node
            
    def _has_edge(self, source: str, target: str, key: Optional[str]=None):
        if key:
            return self._G.has_edge(source, target, key=key)
        else:
            return self._G.has_edge(source, target)
        
    def _get_edge_iterator(self, source: str, target: str):
        edges = self._G.get_edge_data(source, target)                
        for key, attributes in edges.items():
            yield key, attributes
            
    def _set_edge_data(self, source: str, target: str, edge_key: str, **kwargs):
        for key, value in kwargs.items():
            self._G[source][target][edge_key][key] = value
            
    def _remove_edge(self, node_x: str, node_y: str, key: str):
        self._G.remove_edge(node_x, node_y, key)
    
    def _get_edge_data(self, source: str, target: str, edge_key: str, key: str):
        return self._G[source][target][edge_key][key]
    
    def _add_edge(
        self, 
        source: str, 
        target: str, 
        code: str, 
        edge_type: int,
        lineno: Optional[int]=None, 
        col_offset: Optional[int]=None, 
        end_lineno: Optional[int]=None, 
        end_col_offset: Optional[int]=None
    ):
        if source and target and source != target and len(code) > 0:
            key = f"{source}_{target}_{code}"

            if self._has_edge(source, target, key=key):
                edge_count = self._get_edge_data(source, target, key, "count")
                self._set_edge_data(source, target, key, count=edge_count + 1)
            else:
                position = {
                    "lineno": lineno,
                    "col_offset": col_offset,
                    "end_lineno": end_lineno,
                    "end_col_offset": end_col_offset,
                }
                self._G.add_edge(source, target, code=code, key=key, edge_type=edge_type, count=1, **position)
        else:
            self._logger.debug(f"Ignoring edge {source} -> {target} with code {code}")