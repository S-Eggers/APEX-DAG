import re
import ast
import networkx as nx
from logging import Logger
from typing import Optional

from ApexDAG.util.draw import Draw
from ApexDAG.state import Stack, State
from ApexDAG.util.logging import setup_logging
from ApexDAG.sca import (
    NODE_TYPES,
    EDGE_TYPES,
    VERBOSE,
    ASTGraph,
    get_operator_description,
    flatten_list,
    convert_multidigraph_to_digraph,
    get_subgraph,
    save_graph,
    load_graph
)


class PythonDataFlowGraph(ASTGraph, ast.NodeVisitor):
    def __init__(self, notebook_path: str = "") -> None:
        super().__init__()
        self._logger: Logger = setup_logging(f"py_data_flow_graph {notebook_path}", VERBOSE)
        self._state_stack: Stack = Stack()
        self._current_state: State = self._state_stack.get_current_state()

    def visit_Import(self, node: ast.Import) -> ast.Import:
        # Track imported modules
        for alias in node.names:
            self._state_stack.imported_names[alias.asname or alias.name] = alias.name

        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        # Track imported modules from 'import from' statements
        module = node.module
        for alias in node.names:
            self._state_stack.imported_names[alias.asname or alias.name] = module
            self._state_stack.import_from_modules[alias.asname or alias.name] = module

        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        target = node.targets[0]
        value = node.value

        # Get the versioned name of the target variable using the line number
        target_names = self._get_names(target)
        if not target_names:
            code = ast.get_source_segment(self.code, target)
            self._logger.error("Could not get target names for %s", code)
            self._logger.debug(code)
            target_names = []

        target_names = flatten_list(target_names)
        for target_name_ in target_names:
            is_obj_attribute = isinstance(target_name_, list)
            target_name = target_name_[0] if is_obj_attribute else target_name_

            # Create a new version for this variable
            self._current_state.set_current_variable(self._get_versioned_name(target_name, node.lineno))
            self._current_state.set_current_target(target_name)

            if isinstance(value, ast.Lambda):
                base_name = self._get_base_name(target)
                self._state_stack.functions[base_name] = {"node": self._current_state.current_variable}
                self._current_state.add_node(self._current_state.current_variable, NODE_TYPES["INTERMEDIATE"])
            else:
                self._current_state.add_node(self._current_state.current_variable, NODE_TYPES["VARIABLE"])

            value.parent = node
            self.visit(value)

            # Update the latest version of this variable
            if target_name not in self._current_state.variable_versions:
                self._current_state.variable_versions[target_name] = []
                previous_version = None
            else:
                previous_version = self._get_last_variable_version(target_name)

            self._current_state.variable_versions[target_name].append(self._current_state.current_variable)

            if previous_version and not self._current_state.has_edge(previous_version, self._current_state.current_variable):
                self._current_state.add_edge(
                    previous_version,
                    self._current_state.current_variable,
                    "reassign", 
                    EDGE_TYPES["OMITTED"],
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset
                )

        self._current_state.set_current_variable(None)
        self._current_state.set_current_target(None)
        self._current_state.set_last_variable(None)
        self._current_state.edge_for_current_target = {}

        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign:
        # get operator
        try:
            operator =  node.op.__class__.__name__.lower()
        except AttributeError:
            code = ast.get_source_segment(self.code, node)
            self._logger.debug("Could not get operator for %s", code)
            operator = None

        target = node.target
        base_name = self._get_base_name(target)
        current_target = self._get_versioned_name(base_name, node.lineno)
        self._current_state.add_node(current_target, NODE_TYPES["VARIABLE"])
        self._current_state.add_edge(
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
        return node
    
    def visit_AnnAssign(self, node) -> ast.AnnAssign:
        node.targets = [node.target]
        self.visit_Assign(node)
        return node


    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        base_name = self._get_base_name(node.value)
        # just a name as an expression e.g. x, is not a data flow
        is_call = isinstance(node.value, ast.Call)
        is_first_order = is_call and isinstance(node.value.func, ast.Name)
        is_self_defined = base_name in self._state_stack.functions
        is_imported = base_name in self._state_stack.import_from_modules or base_name in self._state_stack.imported_names

        if base_name and (not is_first_order or is_self_defined or is_imported):
            self._current_state.set_current_variable(self._get_versioned_name(base_name, node.lineno))
            self._current_state.set_current_target(base_name)
            self._current_state.add_node(self._current_state.current_variable, NODE_TYPES["VARIABLE"])

            node.value.parent = node
            self.visit(node.value)

            # Update the latest version of this variable
            if base_name not in self._current_state.variable_versions:
                self._current_state.variable_versions[base_name] = []

            self._current_state.variable_versions[base_name].append(self._current_state.current_variable)

        self._current_state.set_current_variable(None)
        self._current_state.set_current_target(None)
        self._current_state.set_last_variable(None)
        self._current_state.edge_for_current_target = {}

        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        # ToDo: this method is executed twice
        if hasattr(node, "parent") and isinstance(node.parent, ast.Assign):
            name = self._get_names(node)[0]
            name = self._get_versioned_name(name, node.lineno)
            parent_name = self._get_names(node.parent.targets[0])[0]            
            context_name = f"{parent_name}_{name}"
            if context_name in self._state_stack:
                return node

            self._state_stack.functions[parent_name]["context"] = context_name            
            self._state_stack.functions[parent_name]["args"] = self._process_arguments(node.args)
            self._state_stack.functions[parent_name]["is_recursive"] = False
            parent_context = self._current_state.context
            self._state_stack.create_child_state(context_name, parent_context)
            self._current_state = self._state_stack.get_current_state()

            for arg in self._state_stack.functions[parent_name]["args"]["args"]:
                argument_node = f"{arg}_{node.lineno}"
                self._current_state.add_node(argument_node, NODE_TYPES["VARIABLE"])
                self._current_state.variable_versions[arg] = [argument_node]            

            self.visit(node.body)
            self._state_stack.restore_state(parent_context)
            self._current_state = self._state_stack.get_current_state()
            
            # self._current_state.add_node(context_name, NODE_TYPES["INTERMEDIATE"])

        else:
            code = ast.get_source_segment(self.code, node)
            message = f"Ignoring lambda function {code} as it is not assigned to a variable"
            self._logger.debug(message)
            self._logger.debug(ast.dump(node))
            super().generic_visit(node)
            arguments = self._process_arguments(node.args)
            for arg in arguments["args"]:
                argument_node = f"{arg}_{node.lineno}"
                self._current_state.add_node(argument_node, NODE_TYPES["VARIABLE"])
                self._current_state.variable_versions[arg] = [argument_node]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        function_name = self._get_names(node)[0]
        context_name = self._get_versioned_name(function_name, node.lineno)
        self._state_stack.functions[function_name] = {
            "node": context_name, 
            "context": context_name,
            "is_recursive": self._check_resursion(node),
            "kwargs": True if node.args.kwarg else False,
            "vararg": True if node.args.vararg else False,
        }
        self._state_stack.functions[function_name]["args"] = self._process_arguments(node.args)

        parent_context = self._current_state.context
        self._state_stack.create_child_state(context_name, parent_context)
        self._current_state = self._state_stack.get_current_state()

        for arg in self._state_stack.functions[function_name]["args"]["args"]:
            argument_node = f"{arg}_{node.lineno}"
            self._current_state.add_node(argument_node, NODE_TYPES["VARIABLE"])
            self._current_state.variable_versions[arg] = [argument_node]

        for stmt in node.body:
            stmt.parent = node
            self.visit(stmt)

        self._state_stack.restore_state(parent_context)
        self._current_state = self._state_stack.get_current_state()

        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
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
        if caller_object_name in self._state_stack.imported_names or caller_object_name in self._state_stack.import_from_modules:
            self._process_library_call(node, caller_object_name, function_name)
        # we are calling a user defined class, e.g. dataframe = DataFrame(), ToDo: currently only working for Constructors
        elif caller_object_name in self._state_stack.classes or function_name in self._state_stack.classes:
            self._process_class_call(node, caller_object_name, function_name)
        # we are calling a user defined function, e.g. a lambda function stored in a variable
        elif not caller_object_name and function_name in self._state_stack.functions:
            self._process_function_call(node, function_name)
        # we are calling a method of a object, e.g. dataframe = dataframe.dropna()
        elif self._current_state.current_target:
            self._process_method_call(node, caller_object_name, function_name)
        else:
            code = ast.get_source_segment(self.code, node)
            self._logger.debug("Ignoring function call %s as it contains no data flow", code)
            self._logger.debug(ast.dump(node))
            self._logger.debug("Caller object name %s not in %s", caller_object_name, self._current_state.current_target)

        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
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
            code = ast.get_source_segment(self.code, node)
            self._logger.debug("Could not get operator for %s", code)
            operator = None

        # Add edges from operands to the new variable version
        if self._current_state.current_variable and operator:
            if left_var:
                left_version = self._get_last_variable_version(left_var)
                self._current_state.add_edge(
                    left_version,
                    self._current_state.current_variable,
                    operator,
                    EDGE_TYPES["CALLER"],
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset
                )
            if right_var:
                right_version = self._get_last_variable_version(right_var)
                self._current_state.add_edge(
                    right_version,
                    self._current_state.current_variable,
                    operator,
                    EDGE_TYPES["CALLER"],
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset
                )

        return node
                
    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        left_var, right_var = self._get_lr_values(node.left, node.comparators[0])
        node.left.parent = node
        node.comparators[0].parent = node

        self.visit(node.left)
        self.visit(node.comparators[0])

        # get operator
        operator = get_operator_description(node)
        if not operator:
            code = ast.get_source_segment(self.code, node)
            self._logger.debug("Could not get operator for %s", code)

        if self._current_state.current_variable and operator:
            if left_var:
                left_version = self._get_last_variable_version(left_var)
                self._current_state.add_edge(
                    left_version,
                    self._current_state.current_variable,
                    operator,
                    EDGE_TYPES["CALLER"],
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset
                )
            if right_var:
                right_version = self._get_last_variable_version(right_var)
                self._current_state.add_edge(
                    right_version,
                    self._current_state.current_variable,
                    operator,
                    EDGE_TYPES["CALLER"],
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset
                )

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if not isinstance(node.ctx, ast.Load):
            return node

        var_name = node.id
        if var_name in self._state_stack.imported_names or var_name in self._state_stack.import_from_modules:
            self._logger.debug("Processing library attribute %s", var_name)
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

            if self._current_state.current_variable:
                self._current_state.add_edge(
                    var_version,
                    self._current_state.current_variable,
                    code_segment,
                    edge_type,
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset
                )
                self._current_state.set_last_variable(var_version)

        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        if hasattr(node, "value"):
            node.value.parent = node
            self.visit(node.value)

        var_name = None
        if self._current_state.last_variable:
            var_name = self._current_state.last_variable

        self._current_state.add_edge(
            var_name,
            self._current_state.current_variable,
            self._tokenize_method(node.attr),
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )

        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.IfExp:
        self._current_state.add_edge(
            self._get_last_variable_version(self._current_state.current_target),
            self._current_state.current_variable,
            "if",
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )
        # visit if expression
        self.visit(node.body)
        # visit else expression
        self.visit(node.orelse)

        return node

    def visit_If(self, node: ast.If) -> ast.If:
        if_branch = True
        # first check if we are in an if or else if statement
        if node.parent and isinstance(node.parent, ast.If) and isinstance(node.parent.orelse[0], ast.If):
            # we are actually in an else if statement
            parent_context = node.parent.parent_context
            if_context = f"else_if_{node.lineno}"
            self._visit_if_body(node.body, if_context, parent_context)
            self._state_stack.branches.append((self._current_state, "else if", EDGE_TYPES["BRANCH"]))
            if_branch = False
            previous_target = None
        # we are not visiting an elif statement (if or else body)
        else:
            node_name = self._get_names(node)[0]
            var_version = self._get_versioned_name(node_name, node.lineno)
            previous_target = self._current_state.current_target
            self._current_state.current_target = var_version
            parent_context = self._current_state.context

            if_context = f"{var_version}_if"
            self._visit_if_body(node.body, if_context, parent_context)
            self._state_stack.branches.append((self._current_state, "if", EDGE_TYPES["BRANCH"]))

        # visit elif statements
        if node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node.parent_context = self._current_state.parent_context
            node.orelse[0].parent = node
            self.visit(node.orelse[0])
        # visit else statement
        elif len(node.orelse) > 0:
            else_context = f"else_{node.lineno}"
            self._visit_if_body(node.orelse, else_context, parent_context)
            self._state_stack.branches.append((self._current_state, "else", EDGE_TYPES["BRANCH"]))
        # merge state
        if if_branch:
            self._state_stack.merge_states(parent_context, self._state_stack.branches)
            self._current_state = self._state_stack.get_current_state()
            self._state_stack.branches = []
            self._current_state.current_target = previous_target

        return node

    def visit_While(self, node: ast.While) -> ast.While:
        node_name = self._get_names(node)[0]
        var_version = self._get_versioned_name(node_name, node.lineno)

        parent_context = self._current_state.context
        while_context = f"{var_version}_while"
        self._state_stack.create_child_state(while_context, parent_context)
        self._current_state = self._state_stack.get_current_state()
        for stmt in node.body:
            stmt.parent = node
            self.visit(stmt)
        contexts = [(self._current_state, "loop", EDGE_TYPES["LOOP"])]

        if node.orelse and len(node.orelse) > 0:
            else_context = f"{var_version}_else"
            self._state_stack.create_child_state(else_context, parent_context)
            self._current_state = self._state_stack.get_current_state()
            for stmt in node.orelse:
                stmt.parent = node
                self.visit(stmt)
            contexts.append((self._current_state, "else", EDGE_TYPES["BRANCH"]))

        self._state_stack.merge_states(parent_context, contexts)
        self._current_state = self._state_stack.get_current_state()
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        node_name = self._get_names(node)[0]
        var_version = self._get_versioned_name(node_name, node.lineno)

        parent_context = self._current_state.context
        for_context = f"{var_version}_for"
        self._state_stack.create_child_state(for_context, parent_context)
        self._current_state = self._state_stack.get_current_state()
        for stmt in node.body:
            stmt.parent = node
            self.visit(stmt)
        contexts = [(self._current_state, "loop", EDGE_TYPES["LOOP"])]

        if node.orelse and len(node.orelse) > 0:
            else_context = f"{var_version}_else"
            self._state_stack.create_child_state(else_context, parent_context)
            self._current_state = self._state_stack.get_current_state()
            for stmt in node.orelse:
                stmt.parent = node
                self.visit(stmt)
            contexts.append((self._current_state, "else", EDGE_TYPES["BRANCH"]))

        self._state_stack.merge_states(parent_context, contexts)
        self._current_state = self._state_stack.get_current_state()
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        name = node.name
        self._state_stack.classes[name] = [name]
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self._state_stack.classes[name].append(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self._state_stack.classes[name].append(target.id)
                    elif isinstance(target, ast.Attribute):
                        self._state_stack.classes[name].append(target.attr)

        return node

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            child.parent = node

        super().generic_visit(node)

    def draw_all_subgraphs(self) -> None:
        for variable in self._current_state.variable_versions:
            self.draw(variable, variable)
        # draw the full graph
        self.draw()

    def draw(self, save_path: str=None, start_node: str=None) -> None:
        draw = Draw(NODE_TYPES, EDGE_TYPES)

        if start_node:
            G_copy = self._current_state.copy_graph()
            self._current_state.set_graph(
                get_subgraph(self._current_state.get_graph(), self._current_state.variable_versions, start_node)
            )
            G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
            draw.dfg(G, save_path)
            self._current_state.set_graph(G_copy)
        else:
            print("Drawing full graph")
            G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
            print("Drawing full graph")
            draw.dfg(G, save_path)

    def webrender(self, save_path: str = None) -> None:
        draw = Draw(NODE_TYPES, EDGE_TYPES)
        G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
        draw.dfg_webrendering(G, save_path)

    def to_json(self) -> str:
        draw = Draw(NODE_TYPES, EDGE_TYPES)
        G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
        return draw.dfg_to_json(G)

    def save_dfg(self, path: str) -> None:
        G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
        save_graph(G, path)

    def read_dfg(self, path: str) -> None:
        self._current_state.set_graph(load_graph(path))
        self._logger.info("Graph successfully loaded from %s", path)

    def optimize(self) -> None:
        self._current_state.optimize()

    def get_graph(self) -> nx.Graph:
        return self._current_state.get_graph()

    def _check_resursion(self, node: ast.FunctionDef) -> bool:
        function_name = node.name
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                if sub_node.func.id == function_name:
                    return True
        return False

    def _process_arguments(self, node: ast.arguments) -> dict[str, list]:
        # posonlyargs, kwonlyargs, vararg, kwarg probably not needed
        arguments = {"args": [], "defaults": []}
        for arg in node.args:
            arguments["args"].append(arg.arg)

        for default in node.defaults:
            arguments["defaults"].append(default)

        return arguments

    def _visit_if_body(self, body: list[ast.AST], context: str, parent_context: str) -> None:
        self._state_stack.create_child_state(context, parent_context)
        self._current_state = self._state_stack.get_current_state()
        for stmt in body:
            stmt.parent = context
            self.visit(stmt)

    def _process_method_call(self, node: ast.Call, caller_object_name: str, tokens: Optional[str]) -> None:
        previous_version = self._get_last_variable_version(caller_object_name)
        previous_version = previous_version if previous_version else self._current_state.last_variable
        tokens = self._tokenize_method(tokens)
        self._current_state.add_edge(
            previous_version,
            self._current_state.current_variable, 
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
                    self._current_state.add_edge(
                        arg_version,
                        self._current_state.current_variable,
                        code_segment,
                        EDGE_TYPES["INPUT"],
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset
                    )
            if isinstance(arg, (ast.Tuple)):
                arg_names = self._get_names(arg)
                arg_names = [arg_name[0] for arg_name in arg_names if arg_name]

                for arg_name in arg_names:  
                    if arg_name:
                        arg_version = self._get_last_variable_version(arg_name)
                        code_segment = self._tokenize_method(arg_name)
                        self._current_state.add_edge(
                            arg_version,
                            self._current_state.current_variable,
                            code_segment,
                            EDGE_TYPES["INPUT"],
                            node.lineno,
                            node.col_offset,
                            node.end_lineno,
                            node.end_col_offset
                        )
                
            
    def _process_library_call(self, node: ast.Call, caller_object_name: str, tokens: str=None) -> None:
        # Add the import node and connect it
        import_node = self._state_stack.imported_names[caller_object_name]
        self._current_state.add_node(import_node, NODE_TYPES["IMPORT"])
        tokens = tokens if tokens else ast.get_source_segment(self.code, node.func)
        tokens = self._tokenize_method(tokens)
        self._current_state.add_edge(
            import_node,
            self._current_state.current_variable,
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
                    self._current_state.add_edge(
                        arg_version,
                        self._current_state.current_variable,
                        code_segment,
                        EDGE_TYPES["INPUT"],
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset
                    )

        self._current_state.set_last_variable(import_node)

    def _process_class_call(self, node: ast.Call, caller_object_name: str, tokens: str=None) -> None:
        # Add the import node and connect it          
        class_node = self._state_stack.classes[caller_object_name][0]
        self._current_state.add_node(class_node, NODE_TYPES["CLASS"])
        tokens = tokens if tokens else ast.get_source_segment(self.code, node.func)
        tokens = self._tokenize_method(tokens)
        self._current_state.add_edge(
            class_node,
            self._current_state.current_variable,
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
                    self._current_state.add_edge(
                        arg_version,
                        self._current_state.current_variable,
                        code_segment,
                        EDGE_TYPES["INPUT"],
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset
                    )

        self._current_state.set_last_variable(class_node)

    def _process_function_call(self, node: ast.Call, tokens: str=None) -> None:
        function_name = tokens
        if self._state_stack.functions[function_name]["is_recursive"]:
            return

        function_context = self._state_stack.functions[function_name]["context"]
        function_name_tokens = self._tokenize_method(function_name)

        arg_copy: list = self._state_stack.functions[function_name]["args"]["args"].copy()
        mapping: dict = {}
        unprocessable_keywords: list = []
        for keyword in node.keywords:
            mapping[self._get_base_name(keyword.value)] = keyword.arg
            if keyword.arg in arg_copy:
                arg_copy.remove(keyword.arg)
            # dynamically unpacking dictionary arguments
            elif isinstance(keyword.value, ast.Dict):
                for key, value in zip(keyword.value.keys, keyword.value.values):
                    key_base_name = self._get_base_name(key)
                    value_base_name = self._get_base_name(value)
                    if key_base_name in arg_copy:
                        arg_copy.remove(key_base_name)
                        mapping[key_base_name] = value_base_name
                    else:
                        unprocessable_keywords.append(value_base_name)
            elif not keyword.arg:
                base_name = self._get_base_name(keyword.value)
                unprocessable_keywords.append(base_name)
            elif self._state_stack.functions[function_name]["kwargs"]:
                mapping[keyword.arg] = "kwargs"
            elif keyword.arg is None:
                unprocessable_keywords[keyword] = keyword.value
            else:
                mapping[keyword] = "args"

        arg_mapping: dict = {}
        for index, arg in enumerate(node.args):
            arg_mapping[self._get_base_name(arg)] = index

        for keyword in [*list(arg_mapping.keys()), *unprocessable_keywords]:
            if len(arg_copy) > 0:
                mapping[keyword] = arg_copy.pop(0)
            elif self._state_stack.functions[function_name]["kwargs"]:
                mapping[keyword] = "kwargs"
            else:
                mapping[keyword] = "args"

        current_context = self._current_state.context
        self._state_stack.restore_state(function_context)
        self._current_state = self._state_stack.get_current_state()

        for key, value in mapping.items():
            if key in self._current_state.variable_versions:
                self._current_state.variable_versions[value] = self._current_state.variable_versions[key]
                del self._current_state.variable_versions[key]

        self._state_stack.merge_states(current_context, [(self._current_state, function_name_tokens, EDGE_TYPES["FUNCTION_CALL"])])
        self._current_state = self._state_stack.get_current_state()

    def _process_library_attr(self, node: ast.Attribute, caller_object_name: str) -> None:
        # Add the import node and connect it
        import_node = self._state_stack.imported_names[caller_object_name]
        self._current_state.add_node(import_node, NODE_TYPES["IMPORT"])
        tokens = self._tokenize_method(ast.get_source_segment(self.code, node))
        self._current_state.add_edge(
            import_node,
            self._current_state.current_variable,
            tokens,
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )
        self._current_state.set_last_variable(import_node)

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
            code = ast.get_source_segment(self.code, node)
            self._logger.debug("Could not process subscript %s", code)
            self._logger.debug(ast.dump(node))
            code_segment = ast.get_source_segment(self.code, node.parent)
            edge_type = EDGE_TYPES["INPUT"]

        return code_segment, edge_type

    def _get_caller_object(self, value: ast.AST) -> str:
        names = self._get_names(value)

        if names and (
            names[0] in self._state_stack.imported_names or
            names[0] in self._state_stack.import_from_modules or
            names[0] in self._current_state.variable_versions or
            names[0] in self._state_stack.functions or
            names[0] in self._state_stack.classes
        ):
            return names[0]

        return self._current_state.current_target

    def _get_versioned_name(self, var_name: str, lineno: int) -> str:
        return f"{var_name}_{lineno}"

    def _get_base_name(self, node: ast.AST):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return node.value
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
                return names
            case ast.FunctionDef():
                return [node.name]
            case ast.Subscript():
                return self._get_names(node.value)
            case ast.Call():
                return self._get_names(node.func)
            case (ast.If() | ast.IfExp()):
                return ["If"]
            case (ast.While() | ast.For()):
                return ["Loop"]
            case ast.Lambda():
                return ["Lambda"]
            # no error, however, we dont want to process them (probably within Expr aka having a look into the variable context)
            case (ast.BinOp() | ast.Compare() | ast.BoolOp() | ast.UnaryOp() | ast.IfExp() | ast.JoinedStr() | ast.Constant()):
                return None
            case _:
                code = ast.get_source_segment(self.code, node)
                self._logger.error("Could not get names for %s", code)
                self._logger.debug(ast.dump(node))
                return None

    def _tokenize_method(self, method: str) -> str:
        split_by_period = method.split(".")

        # remove library alias as we dont want to learn from it
        for library_alias in self._state_stack.imported_names.keys():
            if library_alias in split_by_period and library_alias not in self._state_stack.import_from_modules:
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
        if variable in self._current_state.variable_versions:
            return self._current_state.variable_versions.get(variable)[-1]

        if self._current_state.parent_context and max_depth > 0:
            current_context = self._current_state.context
            self._state_stack.restore_parent_state()
            self._current_state = self._state_stack.get_current_state()
            variable_version = self._get_last_variable_version(variable, max_depth=max_depth - 1)
            self._state_stack.restore_state(current_context)
            self._current_state = self._state_stack.get_current_state()
            return variable_version
        else:
            return None

    def _import_accessible(self, name: str, max_depth: int = 99) -> bool:
        if name in self._state_stack.import_from_modules or \
            name in self._state_stack.imported_names:
            return True

        if self._current_state.parent_context and max_depth > 0:
            current_context = self._current_state.context
            self._state_stack.restore_parent_state()
            self._current_state = self._state_stack.get_current_state()
            reachable = self._import_accessible(name, max_depth=max_depth - 1)
            self._state_stack.restore_state(current_context)
            self._current_state = self._state_stack.get_current_state()
            return reachable

        return False

    def _class_accessible(self, name: str, max_depth: int = 99) -> bool:
        if name in self._state_stack.classes:
            return True

        if self._current_state.parent_context and max_depth > 0:
            current_context = self._current_state.context
            self._state_stack.restore_parent_state()
            self._current_state = self._state_stack.get_current_state()
            reachable = self._class_accessible(name, max_depth=max_depth - 1)
            self._state_stack.restore_state(current_context)
            self._current_state = self._state_stack.get_current_state()
            return reachable
        
        return False

    def _function_accessible(self, name: str, max_depth: int = 99) -> bool:
        if name in self._state_stack.functions:
            return True

        if self._current_state.parent_context and max_depth > 0:
            current_context = self._current_state.context
            self._state_stack.restore_parent_state()
            self._current_state = self._state_stack.get_current_state()
            reachable = self._function_accessible(name, max_depth=max_depth - 1)
            self._state_stack.restore_state(current_context)
            self._current_state = self._state_stack.get_current_state()
            return reachable

        return False

    def _get_lr_values(self, left: ast.AST, right: ast.AST) -> tuple[Optional[str], Optional[str]]:
        left_var_ = self._get_names(left)
        left_var_ = flatten_list(left_var_) if left_var_ else None
        left_var = left_var_[0] if isinstance(left_var_, list) else left_var_

        right_var_ = self._get_names(right)
        right_var_ = flatten_list(right_var_) if right_var_ else None
        right_var = right_var_[0] if isinstance(right_var_, list) else right_var_

        return left_var, right_var
