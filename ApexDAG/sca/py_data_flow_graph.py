import ast
import re
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
    load_graph,
    CallInliner
)


class PythonDataFlowGraph(ASTGraph, ast.NodeVisitor):
    def __init__(self, notebook_path: str = "", replace_dataflow: bool = False) -> None:
        super().__init__()
        self._replace_dataflow = replace_dataflow
        self._logger: Logger = setup_logging(
            f"py_data_flow_graph {notebook_path}", VERBOSE
        )
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
        value = node.value
        assignment_code = ast.get_source_segment(self.current_cell_source, node) or ""

        for target in node.targets:
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
                self._current_state.set_current_variable(
                    self._get_versioned_name(target_name, node.lineno)
                )
                self._current_state.set_current_target(target_name)

                if isinstance(value, ast.Lambda):
                    base_name = self._get_base_name(target)
                    self._state_stack.functions[base_name] = {
                        "node": self._current_state.current_variable,
                        "context": None,
                        "args": {"args": [], "defaults": []},
                        "is_recursive": False,
                        "kwargs": False,
                        "vararg": False,
                        "return_nodes": [],
                    }
                    self._add_node(
                        self._current_state.current_variable, 
                        NODE_TYPES["INTERMEDIATE"] if isinstance(value, ast.Lambda) else NODE_TYPES["VARIABLE"], 
                        code=assignment_code
                    )
                else:
                    self._add_node(
                        self._current_state.current_variable, 
                        NODE_TYPES["VARIABLE"], 
                        code=assignment_code
                    )

                value.parent = node
                self.visit(value)

                # Update the latest version of this variable
                if target_name not in self._current_state.variable_versions:
                    self._current_state.variable_versions[target_name] = []
                    previous_version = None
                else:
                    previous_version = self._get_last_variable_version(target_name)

                self._current_state.variable_versions[target_name].append(
                    self._current_state.current_variable
                )

                if previous_version and not self._current_state.has_edge(
                    previous_version, self._current_state.current_variable
                ):
                    self._add_edge(
                        previous_version,
                        self._current_state.current_variable,
                        assignment_code,
                        EDGE_TYPES["OMITTED"],
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset,
                    )

        self._current_state.set_current_variable(None)
        self._current_state.set_current_target(None)
        self._current_state.set_last_variable(None)
        self._current_state.edge_for_current_target = {}

        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign:
        target = node.target
        value = node.value
        operator = node.op.__class__.__name__.lower()
        aug_code = ast.get_source_segment(self.current_cell_source, node) or ""

        target_base_name = self._get_base_name(target)
        new_target_version = self._get_versioned_name(target_base_name, node.lineno)
        self._add_node(new_target_version, NODE_TYPES["VARIABLE"], code=aug_code)

        old_target_version = self._get_last_variable_version(target_base_name)
        if old_target_version:
            self._add_edge(
                old_target_version,
                new_target_version,
                operator,
                EDGE_TYPES["CALLER"],
                raw_code=aug_code,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
            )
        original_current_var = self._current_state.current_variable
        self._current_state.set_current_variable(new_target_version)
        self.visit(value)
        self._current_state.set_current_variable(original_current_var)

        # Update variable version tracking
        if target_base_name not in self._current_state.variable_versions:
            self._current_state.variable_versions[target_base_name] = []
        self._current_state.variable_versions[target_base_name].append(
            new_target_version
        )

        # Reset state
        self._current_state.set_current_variable(None)

        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        # ToDo: we might want to do something with the annotation
        if not hasattr(node, "value") or not node.value:
            return node

        assign_node = ast.Assign(targets=[node.target], value=node.value)
        ast.copy_location(assign_node, node)
        self.visit_Assign(assign_node)
        return node

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if not node.value:
            return node

        parent = getattr(node, "parent", None)
        function_def_node = None
        while parent:
            if isinstance(parent, ast.FunctionDef):
                function_def_node = parent
                break
            parent = getattr(parent, "parent", None)

        if function_def_node:
            function_name = self._get_names(function_def_node)[0]

            return_node_name = f"return_{function_name}_{node.lineno}"
            self._add_node(return_node_name, NODE_TYPES["INTERMEDIATE"])

            if "return_nodes" in self._state_stack.functions[function_name]:
                self._state_stack.functions[function_name]["return_nodes"].append(
                    return_node_name
                )

            original_variable = self._current_state.current_variable
            self._current_state.set_current_variable(return_node_name)
            self.visit(node.value)
            self._current_state.set_current_variable(original_variable)

        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        expr_code = ast.get_source_segment(self.current_cell_source, node) or ""
        base_name = self._get_base_name(node.value)
        # just a name as an expression e.g. x, is not a data flow
        is_call = isinstance(node.value, ast.Call)
        is_first_order = is_call and isinstance(node.value.func, ast.Name)
        is_self_defined = base_name in self._state_stack.functions
        is_imported = (
            base_name in self._state_stack.import_from_modules
            or base_name in self._state_stack.imported_names
        )

        if base_name and (not is_first_order or is_self_defined or is_imported):
            # check for prior version of the variable
            prior_versions = self._current_state.variable_versions.get(base_name, [])
            if prior_versions:
                last_version = prior_versions[-1]
                self._current_state.set_current_variable(last_version)
            else:
                self._current_state.set_current_variable(
                    self._get_versioned_name(base_name, node.lineno)
                )
                self._add_node(
                    self._current_state.current_variable, 
                    NODE_TYPES["VARIABLE"],
                    code=expr_code
                )
            self._current_state.set_current_target(base_name)
            node.value.parent = node
            self.visit(node.value)

            # Update the latest version of this variable
            if base_name not in self._current_state.variable_versions:
                self._current_state.variable_versions[base_name] = []

            self._current_state.variable_versions[base_name].append(
                self._current_state.current_variable
            )

        self._current_state.set_current_variable(None)
        self._current_state.set_current_target(None)
        self._current_state.set_last_variable(None)
        self._current_state.edge_for_current_target = {}

        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        if hasattr(node, "_visited"):
            return node
        node._visited = True

        if isinstance(getattr(node, "parent", None), ast.Assign):
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
            
            with self._state_stack.scope(context_name, parent_context) as scoped_state:
                self._current_state = scoped_state

                for arg in self._state_stack.functions[parent_name]["args"]["args"]:
                    argument_node = f"{arg}_{node.lineno}"
                    self._add_node(argument_node, NODE_TYPES["VARIABLE"])
                    self._current_state.variable_versions[arg] = [argument_node]

                self.visit(node.body)
                
            self._current_state = self._state_stack.get_current_state()

        else:
            code = ast.get_source_segment(self.code, node)
            message = f"Ignoring lambda function {code} as it is not assigned to a variable"
            self._logger.debug(message)
            self._logger.debug(ast.dump(node))
            super().generic_visit(node)
            arguments = self._process_arguments(node.args)
            for arg in arguments["args"]:
                argument_node = f"{arg}_{node.lineno}"
                self._add_node(argument_node, NODE_TYPES["VARIABLE"])
                self._current_state.variable_versions[arg] = [argument_node]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        function_name = self._get_names(node)[0]
        context_name = self._get_versioned_name(function_name, node.lineno)
        self._state_stack.functions[function_name] = {
            "node": context_name,
            "context": context_name,
            "is_recursive": self._check_resursion(node),
            "kwargs": bool(node.args.kwarg),
            "vararg": bool(node.args.vararg),
            "return_nodes": [],
            "parent_class": node.parent.name if isinstance(node.parent, ast.ClassDef) else None,
        }
        self._state_stack.functions[function_name]["args"] = self._process_arguments(node.args)
        self._add_node(context_name, NODE_TYPES["FUNCTION"])

        parent_context = self._current_state.context
        
        with self._state_stack.scope(context_name, parent_context) as scoped_state:
            self._current_state = scoped_state
            
            for arg in self._state_stack.functions[function_name]["args"]["args"]:
                argument_node = f"{arg}_{node.lineno}"
                self._add_node(argument_node, NODE_TYPES["VARIABLE"])
                self._current_state.variable_versions[arg] = [argument_node]

            for stmt in node.body:
                stmt.parent = node
                self.visit(stmt)

        self._current_state = self._state_stack.get_current_state()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.visit_FunctionDef(node)
        return node

    def visit_Await(self, node: ast.Await) -> ast.Await:
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        # Visit all arguments first to ensure any nested calls are processed


        caller_object_name = None
        function_name = None

        if isinstance(node.func, ast.Attribute):
            caller_object_name = self._get_caller_object(node.func.value)
            function_name = node.func.attr
            if hasattr(node.func, "value"):
                node.func.value.parent = node.func
        elif isinstance(node.func, ast.Name):
            caller_object_name = self._get_caller_object(node.func)
            function_name = node.func.id
        elif isinstance(node.func, (ast.Call, ast.Subscript)):
            caller_object_name = self._get_caller_object(node.func)
            function_name = "__call__"
            self.visit(node.func) # Ensure the function itself (if a call or subscript) is visited
        else:
            raise NotImplementedError(
                f"Unsupported function call {ast.get_source_segment(self.code, node)} with node {ast.dump(node)}"
            )

        # we are calling a method of an imported module, e.g. pd.read_csv() or an
        # imported first order method, e.g. read_csv()
        if (
            caller_object_name in self._state_stack.imported_names
            or caller_object_name in self._state_stack.import_from_modules
        ):
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self._process_library_call(node, caller_object_name, function_name)
        # we are calling a user defined class, e.g. dataframe = DataFrame(), ToDo: currently only working for Constructors
        elif (
            caller_object_name in self._state_stack.classes
            or caller_object_name in self._state_stack.instances
            or function_name in self._state_stack.classes
        ):
            is_instance = caller_object_name in self._state_stack.instances
            self._process_class_call(node, caller_object_name, function_name, is_instance)
        # we are calling a user defined function, e.g. a lambda function stored in a variable
        elif (
            not caller_object_name or caller_object_name == function_name
        ) and function_name in self._state_stack.functions:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self._process_function_call(node, function_name)
        elif function_name in [
            "enumerate",
            "zip",
            "next",
            "iter",
            "range",
            "sorted",
            "map",
            "filter",
        ]:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self._process_builtin_call(node, function_name)
        # we are calling a method of a object, e.g. dataframe = dataframe.dropna()
        elif self._current_state.current_target and not (caller_object_name in self._state_stack.instances):
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            # List comp problem otherwise - args would be visited 2 times!!!
            self._process_method_call(node, caller_object_name, function_name)
        else:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            code = ast.get_source_segment(self.code, node)
            self._logger.debug(
                "Ignoring function call %s as it contains no data flow", code
            )
            self._logger.debug(ast.dump(node))
            self._logger.debug(
                "Caller object name %s not in %s",
                caller_object_name,
                self._current_state.current_target,
            )

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
            operator = node.op.__class__.__name__.lower()
        except AttributeError:
            code = ast.get_source_segment(self.code, node)
            self._logger.debug("Could not get operator for %s", code)
            operator = None

        # Add edges from operands to the new variable version
        if self._current_state.current_variable and operator:
            if left_var:
                left_version = self._get_last_variable_version(left_var)
                self._add_edge(
                    left_version,
                    self._current_state.current_variable,
                    operator,
                    EDGE_TYPES["CALLER"],
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset,
                )
            if right_var:
                right_version = self._get_last_variable_version(right_var)
                self._add_edge(
                    right_version,
                    self._current_state.current_variable,
                    operator,
                    EDGE_TYPES["CALLER"],
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset,
                )

        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        node.left.parent = node
        self.visit(node.left)
        left_operand = node.left

        for i, op in enumerate(node.ops):
            right_operand = node.comparators[i]
            right_operand.parent = node
            self.visit(right_operand)

            left_var, right_var = self._get_lr_values(left_operand, right_operand)
            operator = get_operator_description(op)

            if self._current_state.current_variable and operator:
                if left_var:
                    left_version = self._get_last_variable_version(left_var)
                    self._add_edge(
                        left_version,
                        self._current_state.current_variable,
                        operator,
                        EDGE_TYPES["CALLER"],
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset,
                    )
                if right_var:
                    right_version = self._get_last_variable_version(right_var)
                    self._add_edge(
                        right_version,
                        self._current_state.current_variable,
                        operator,
                        EDGE_TYPES["CALLER"],
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset,
                    )

            left_operand = right_operand

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if not isinstance(node.ctx, ast.Load):
            return node

        var_name = node.id
        if (
            var_name in self._state_stack.imported_names
            or var_name in self._state_stack.import_from_modules
        ):
            self._logger.debug("Processing library attribute %s", var_name)
            self._process_library_attr(node, var_name)
            return node
        else:
            var_version = self._get_last_variable_version(var_name)
            is_base_of_attribute = hasattr(node, "parent") and isinstance(
                node.parent, ast.Attribute
            )
            if is_base_of_attribute:
                self._current_state.set_last_variable(var_version)
                return node

            edge_type = EDGE_TYPES["INPUT"]
            # Determine the full code context for more complex structures
            if hasattr(node, "parent") and isinstance(node.parent, ast.Subscript):
                code_segment, edge_type = self._process_subscript(node)
            else:
                code_segment = var_name

            code_segment = self._tokenize_method(code_segment)

            if self._current_state.current_variable:
                self._add_edge(
                    var_version,
                    self._current_state.current_variable,
                    code_segment,
                    edge_type,
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset,
                )
            self._current_state.set_last_variable(None)

        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        if hasattr(node, "value"):
            node.value.parent = node
            self.visit(node.value)

        var_name = None
        if self._current_state.last_variable:
            var_name = self._current_state.last_variable

        self._add_edge(
            var_name,
            self._current_state.current_variable,
            self._tokenize_method(node.attr),
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
        )

        is_part_of_chain = hasattr(node, "parent") and isinstance(
            node.parent, ast.Attribute
        )
        if is_part_of_chain:
            self._current_state.set_last_variable(
                self._current_state.current_variable
            )
        else:
            self._current_state.set_last_variable(None)

        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if self._current_state.current_variable:
            literal_val = str(node.value)
            
            truncated_literal = literal_val
            if len(truncated_literal) > 50:
                truncated_literal = truncated_literal[:47] + "..."

            tokenized_label = self._tokenize_literal(truncated_literal)
            
            cell_context = getattr(self, "current_cell_id", "unknown")
            short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"
            
            literal_node_name = f"literal_{short_cell}_{node.lineno}_{node.col_offset}"
            
            self._add_node(literal_node_name, NODE_TYPES["LITERAL"], code=literal_val)
            
            self._add_edge(
                source=literal_node_name,
                target=self._current_state.current_variable,
                label=tokenized_label,
                edge_type=EDGE_TYPES["INPUT"],
                raw_code=str(node.value),
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
            )
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.IfExp:
        self._add_edge(
            self._get_last_variable_version(self._current_state.current_target),
            self._current_state.current_variable,
            "if",
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
        )
        # visit if expression
        self.visit(node.body)
        # visit else expression
        self.visit(node.orelse)

        return node

    def visit_If(self, node: ast.If) -> ast.If:
        if_branch = True
        is_elif = (
            hasattr(node, "parent")
            and isinstance(node.parent, ast.If)
            and node.parent.orelse
            and node.parent.orelse[0] == node
        )
        parent_context = node.parent_context if is_elif else self._current_state.context
        node.parent_context = parent_context

        if is_elif:
            cell_context = getattr(self, "current_cell_id", "unknown_cell")
            if_context = f"else_if_{cell_context}_{node.lineno}"
            branch_state = self._visit_if_body(node.body, if_context, parent_context, node)
            self._state_stack.branches.append((branch_state, "else if", EDGE_TYPES["BRANCH"]))
            if_branch = False
            previous_target = None
        else:
            node_name = self._get_names(node)[0]
            var_version = self._get_versioned_name(node_name, node.lineno)
            previous_target = self._current_state.current_target
            self._current_state.current_target = var_version

            if_context = f"{var_version}_if"
            branch_state = self._visit_if_body(node.body, if_context, parent_context, node)
            self._state_stack.branches.append((branch_state, "if", EDGE_TYPES["BRANCH"]))

        self._current_state = self._state_stack.get_current_state()

        if node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node.orelse[0].parent = node
            node.orelse[0].parent_context = node.parent_context
            self.visit(node.orelse[0])
        elif len(node.orelse) > 0:
            cell_context = getattr(self, "current_cell_id", "unknown_cell")
            else_context = f"else_{cell_context}_{node.lineno}"
            branch_state = self._visit_if_body(node.orelse, else_context, node.parent_context, node)
            self._state_stack.branches.append((branch_state, "else", EDGE_TYPES["BRANCH"]))
            self._current_state = self._state_stack.get_current_state()

        if if_branch:
            cell_context = getattr(self, "current_cell_id", "unknown_cell")
            self._state_stack.merge_states(node.parent_context, self._state_stack.branches, cell_id=cell_context)
            self._current_state = self._state_stack.get_current_state()
            self._state_stack.branches = []
            self._current_state.current_target = previous_target

        return node

    def visit_While(self, node: ast.While) -> ast.While:
        node_name = self._get_names(node)[0]
        var_version = self._get_versioned_name(node_name, node.lineno)
        parent_context = self._current_state.context
        while_context = f"{var_version}_while"
        
        with self._state_stack.scope(while_context, parent_context) as scoped_state:
            self._current_state = scoped_state
            for stmt in node.body:
                stmt.parent = node
                self.visit(stmt)
            contexts = [(self._current_state, "start_loop", EDGE_TYPES["LOOP"])]
            
        self._current_state = self._state_stack.get_current_state()

        if node.orelse and len(node.orelse) > 0:
            else_context = f"{var_version}_else"
            with self._state_stack.scope(else_context, parent_context) as scoped_state:
                self._current_state = scoped_state
                for stmt in node.orelse:
                    stmt.parent = node
                    self.visit(stmt)
                contexts.append((self._current_state, "else", EDGE_TYPES["BRANCH"]))
                
            self._current_state = self._state_stack.get_current_state()

        cell_context = getattr(self, "current_cell_id", "unknown_cell")
        self._state_stack.merge_states(parent_context, contexts, cell_id=cell_context)
        self._current_state = self._state_stack.get_current_state()
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        temp_iterable_node = self._handle_iterable(node.iter)
        raw_targets = self._get_names(node.target)
        all_targets = self._get_target_components(raw_targets) if raw_targets else []

        parent_context = self._current_state.context
        cell_context = getattr(self, "current_cell_id", "unknown_cell")
        for_context = f"for_loop_{cell_context}_{node.lineno}"
        
        with self._state_stack.scope(for_context, parent_context) as scoped_state:
            self._current_state = scoped_state

            for components in all_targets:
                base_name = components[0]
                target_version = self._get_versioned_name(base_name, node.lineno)
                attribute_path = "." + ".".join(components[1:])
                edge_label = f"iterate into {attribute_path}" if len(components) > 1 else "iterate"

                self._add_node(target_version, NODE_TYPES["VARIABLE"])
                self._current_state.variable_versions[base_name] = [target_version]
                self._add_edge(
                    temp_iterable_node,
                    target_version,
                    edge_label,
                    EDGE_TYPES["LOOP"],
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=node.end_lineno,
                    end_col_offset=node.end_col_offset,
                )

            for stmt in node.body:
                stmt.parent = node
                self.visit(stmt)

            contexts = [(self._current_state, "start_loop", EDGE_TYPES["LOOP"])]
            
        self._current_state = self._state_stack.get_current_state()

        if node.orelse and len(node.orelse) > 0:
            cell_context = getattr(self, "current_cell_id", "unknown_cell")
            else_context = f"for_else_{cell_context}_{node.lineno}"
            with self._state_stack.scope(else_context, parent_context) as scoped_state:
                self._current_state = scoped_state
                for stmt in node.orelse:
                    stmt.parent = node
                    self.visit(stmt)
                contexts.append((self._current_state, "else", EDGE_TYPES["BRANCH"]))
            
            self._current_state = self._state_stack.get_current_state()

        for components in all_targets:
            base_name = components[0]
            if base_name in self._current_state.variable_versions:
                del self._current_state.variable_versions[base_name]

        cell_context = getattr(self, "current_cell_id", "unknown_cell")
        self._state_stack.merge_states(parent_context, contexts, cell_id=cell_context)
        self._current_state = self._state_stack.get_current_state()
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        if hasattr(node, "_visited"):
            return node
        node._visited = True
        
        # Global uniqueness prefix
        cell_context = getattr(self, "current_cell_id", "unknown")
        short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"

        final_list_variable = self._current_state.current_variable
        if not final_list_variable:
            # Inject short_cell
            final_list_variable = f"list_comp_result_{short_cell}_{node.lineno}"
            self._add_node(final_list_variable, NODE_TYPES["VARIABLE"])

        parent_context = self._current_state.context
        if self._state_stack.nested:
            # Inject short_cell
            list_comp_context = f"{parent_context}_nested_list_comp_{short_cell}_{node.lineno}"
        else:
            # Inject short_cell
            list_comp_context = f"list_comp_{short_cell}_{node.lineno}"
            
        with self._state_stack.scope(list_comp_context, parent_context) as scoped_state:
            self._current_state = scoped_state

            start_loop_node = f"{final_list_variable}_start_loop"
            self._add_node(start_loop_node, NODE_TYPES["LOOP"])
            
            # Inject short_cell
            initial_list_node = f"list_comp_start_{short_cell}_{node.lineno}"
            self._add_node(initial_list_node, NODE_TYPES["INTERMEDIATE"])
            
            self._current_state.set_current_variable(start_loop_node)
            self._add_edge(
                source=start_loop_node,
                target=initial_list_node,
                label="start_loop",
                edge_type=EDGE_TYPES["LOOP"],
                raw_code="start_loop",
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
            )

            current_list_version = initial_list_node
            self._add_node(final_list_variable, NODE_TYPES["VARIABLE"])

            for generator in node.generators:
                temp_iterable_node = self._handle_iterable(generator.iter)
                raw_targets = self._get_names(generator.target)
                all_targets = self._get_target_components(raw_targets) if raw_targets else []
                
                for components in all_targets:
                    base_name = components[0]
                    target_version = self._get_versioned_name(
                        base_name, generator.target.lineno
                    )
                    self._add_node(target_version, NODE_TYPES["VARIABLE"])
                    self._current_state.variable_versions[base_name] = [target_version]
                    
                    self._add_edge(
                        source=temp_iterable_node,
                        target=target_version,
                        label="iterate",
                        edge_type=EDGE_TYPES["LOOP"],
                        raw_code="iterate",
                        lineno=generator.target.lineno,
                        col_offset=generator.target.col_offset,
                        end_lineno=generator.target.end_lineno,
                        end_col_offset=generator.target.end_col_offset,
                    )
                    self._add_edge(
                        source=target_version,
                        target=start_loop_node,
                        label="iterate",
                        edge_type=EDGE_TYPES["LOOP"],
                        raw_code="iterate",
                        lineno=generator.target.lineno,
                        col_offset=generator.target.col_offset,
                        end_lineno=generator.target.end_lineno,
                        end_col_offset=generator.target.end_col_offset,
                    )

                for if_cond in generator.ifs:
                    self._current_state.set_current_variable(current_list_version)
                    self.visit(if_cond)
                
                if isinstance(node.elt, ast.ListComp):
                    self._state_stack.nested = True
                    self.visit(node.elt)
                    self._state_stack.nested = False
                
                self._current_state.set_current_variable(None)

            self._add_edge(
                source=current_list_version,
                target=final_list_variable,
                label="end_loop", 
                edge_type=EDGE_TYPES["LOOP"],
                raw_code="end_loop",
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
            )
            self._current_state.set_current_variable(None)

            for generator in node.generators:
                raw_targets = self._get_names(generator.target)
                all_targets = (
                    self._get_target_components(raw_targets) if raw_targets else []
                )
                for components in all_targets:
                    base_name = components[0]
                    if base_name in self._current_state.variable_versions:
                        del self._current_state.variable_versions[base_name]

        self._current_state = self._state_stack.get_current_state()
        
        # Merge back using full cell ID
        cell_context_full = getattr(self, "current_cell_id", "unknown_cell")
        self._state_stack.merge_states(
            parent_context,
            [(self._state_stack._state[list_comp_context], "list_comp", EDGE_TYPES["CALLER"])],
            cell_id=cell_context_full
        )
        self._current_state = self._state_stack.get_current_state()
        del self._state_stack._state[list_comp_context]

        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        name = node.name
        self._state_stack.classes[name] = [name]
        self._current_state.current_class = name
        for stmt in node.body:
            stmt.parent = node
            self.visit(stmt)
            if isinstance(stmt, ast.FunctionDef):
                self._state_stack.classes[name].append(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self._state_stack.classes[name].append(target.id)
                    elif isinstance(target, ast.Attribute):
                        self._state_stack.classes[name].append(target.attr)
        self._current_state.current_class = None
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

    def draw(self, save_path: str = None, start_node: str = None) -> None:
        draw = Draw(NODE_TYPES, EDGE_TYPES)

        if start_node:
            G_copy = self._current_state.copy_graph()
            self._current_state.set_graph(
                get_subgraph(
                    self._current_state.get_graph(),
                    self._current_state.variable_versions,
                    start_node,
                )
            )
            G = convert_multidigraph_to_digraph(
                self._current_state.get_graph(), NODE_TYPES
            )
            draw.dfg(G, save_path)
            self._current_state.set_graph(G_copy)
        else:
            G = convert_multidigraph_to_digraph(
                self._current_state.get_graph(), NODE_TYPES
            )
            # print("Drawing full graph")
            draw.dfg(G, save_path)

    def webrender(self, save_path: str = None) -> None:
        draw = Draw(NODE_TYPES, EDGE_TYPES)
        G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
        draw.dfg_webrendering(G, save_path)

    def set_domain_label(self, attrs: dict, name: str):
        nx.set_edge_attributes(self._current_state._G, attrs, name=name)

    def set_domain_node_label(self, attrs: dict, name: str):
        nx.set_node_attributes(self._current_state._G, attrs, name=name)

    def save_dfg(self, path: str) -> None:
        G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
        save_graph(G, path)

    def read_dfg(self, path: str) -> None:
        self._current_state.set_graph(load_graph(path))
        self._logger.info("Graph successfully loaded from %s", path)

    def optimize(self) -> None:
        self._current_state.optimize()

    def filter_relevant(self, lineage_mode: bool = False) -> None:
        self._current_state.filter_relevant(lineage_mode=lineage_mode)

    def get_graph(self) -> nx.Graph:
        return self._current_state.get_graph()

    def get_edges(self, data: bool = False):
        return self._current_state.get_graph().edges(data=data)

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

    def _visit_if_body(
        self,
        body: list[ast.AST],
        context: str,
        parent_context: str,
        parent_node: ast.AST,
    ) -> "State":
        self._state_stack.create_child_state(context, parent_context)
        self._current_state = self._state_stack.get_current_state()
        for stmt in body:
            stmt.parent = parent_node
            stmt.parent_context = context
            self.visit(stmt)
        return self._current_state

    def _process_method_call(
        self, node: ast.Call, caller_object_name: str, tokens: Optional[str]
    ) -> None:
        previous_version = self._get_last_variable_version(caller_object_name)
        previous_version = (
            previous_version if previous_version else self._current_state.last_variable
        )
        tokens = self._tokenize_method(tokens)
        self._add_edge(
            previous_version,
            self._current_state.current_variable,
            tokens,
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
        )

        for arg in node.args:
            if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                self._process_name_attr_sub_args(node, arg)
            elif isinstance(arg, (ast.Tuple)):
                arg_names = self._get_names(arg)
                if not arg_names:
                    continue

                processed_arg_names = [
                    flatten_list(name_list)[0]
                    for name_list in arg_names
                    if name_list and flatten_list(name_list)
                ]

                for arg_name in processed_arg_names:
                    if arg_name:
                        arg_version = self._get_last_variable_version(arg_name)
                        code_segment = self._tokenize_method(arg_name)
                        self._add_edge(
                            arg_version,
                            self._current_state.current_variable,
                            code_segment,
                            EDGE_TYPES["INPUT"],
                            node.lineno,
                            node.col_offset,
                            node.end_lineno,
                            node.end_col_offset,
                        )
            #else:
            #   self.visit(arg)

    def _process_library_call(
        self, node: ast.Call, caller_object_name: str, tokens: str = None
    ) -> None:
        # Add the import node and connect it
        import_node = self._state_stack.imported_names[caller_object_name]
        self._add_node(import_node, NODE_TYPES["IMPORT"])
        tokens = tokens if tokens else ast.get_source_segment(self.code, node.func)
        tokens = self._tokenize_method(tokens)
        self._add_edge(
            import_node,
            self._current_state.current_variable,
            tokens,
            EDGE_TYPES["FUNCTION_CALL"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
        )

        for arg in node.args:
            if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                self._process_name_attr_sub_args(node, arg)

        self._current_state.set_last_variable(import_node)

    def _process_name_attr_sub_args(self, node: ast.Call, arg: ast.AST) -> None:
        arg_names = self._get_names(arg)
        if not arg_names:
            return
        arg_name = flatten_list(arg_names)[0]

        if arg_name:
            arg_version = self._get_last_variable_version(arg_name)
            code_segment = self._tokenize_method(arg_name)
            self._add_edge(
                arg_version,
                self._current_state.current_variable,
                code_segment,
                EDGE_TYPES["INPUT"],
                node.lineno,
                node.col_offset,
                node.end_lineno,
                node.end_col_offset,
            )

    def _process_class_call(
        self, node: ast.Call, caller_object_name: str, tokens: str = None, is_instance: bool = False
    ) -> None:
        inliner = CallInliner(self)
        inliner.process_class(node, caller_object_name, tokens, is_instance)

    def _process_builtin_call(self, node: ast.Call, function_name: str) -> None:
        if not self._current_state.current_variable:
            return

        builtin_node = "__builtins__"
        self._add_node(builtin_node, NODE_TYPES["IMPORT"])

        self._add_edge(
            builtin_node,
            self._current_state.current_variable,
            function_name,
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
        )

        for arg in node.args:
            arg_names = self._get_names(arg)
            if arg_names:
                base_name = flatten_list(arg_names)[0]
                if base_name and isinstance(base_name, str):
                    arg_version = self._get_last_variable_version(base_name)
                    if arg_version:
                        self._add_edge(
                            arg_version,
                            self._current_state.current_variable,
                            "arg",
                            EDGE_TYPES["INPUT"],
                            arg.lineno,
                            arg.col_offset,
                            arg.end_lineno,
                            arg.end_col_offset,
                        )

    def _process_function_call(self, node: ast.Call, tokens: str = None) -> None:
        inliner = CallInliner(self)
        inliner.process_function(node, tokens)

    def _process_library_attr(
        self, node: ast.Attribute, caller_object_name: str
    ) -> None:
        # Add the import node and connect it
        import_node = self._state_stack.imported_names[caller_object_name]
        self._add_node(import_node, NODE_TYPES["IMPORT"])
        tokens = self._tokenize_method(ast.get_source_segment(self.code, node))
        self._add_edge(
            import_node,
            self._current_state.current_variable,
            tokens,
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
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
            self._import_accessible(names[0])
            or self._class_accessible(names[0])
            or self._function_accessible(names[0])
            or names[0] in self._current_state.variable_versions
        ):
            return names[0]

        return self._current_state.current_target

    def _get_versioned_name(self, var_name: str, lineno: int) -> str:
        cell_context = getattr(self, "current_cell_id", "unknown")
        short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"
        
        return f"{var_name}_{short_cell}_{lineno}"

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
                return ([name[0] + "." + node.attr]) if name else [node.attr]
            case ast.Tuple() | ast.List() | ast.Set():
                names = [
                    self._get_names(elt) for elt in node.elts if self._get_names(elt)
                ]
                return names
            case ast.Dict():
                names = []
                for value in node.values:
                    if value_names := self._get_names(value):
                        names.extend(value_names)
                for key in node.keys:
                    if key_names := self._get_names(key):
                        names.extend(key_names)
                return names
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                if isinstance(node.parent, ast.ClassDef):
                    return [f"{node.parent.name}.{node.name}"]
                return [node.name]
            case ast.Subscript():
                return self._get_names(node.value)
            case ast.Call():
                return self._get_names(node.func)
            case ast.If() | ast.IfExp():
                return ["If"]
            case ast.While() | ast.For():
                return ["Loop"]
            case ast.Lambda():
                return ["Lambda"]
            case ast.Starred():
                return self._get_names(node.value)
            # no error, however, we dont want to process them (probably within Expr aka having a look into the variable context)
            case (
                ast.BinOp()
                | ast.Compare()
                | ast.BoolOp()
                | ast.UnaryOp()
                | ast.IfExp()
                | ast.JoinedStr()
                | ast.Constant()
            ):
                return None
            case _:
                code = ast.get_source_segment(self.code, node)
                self._logger.error("Could not get names for %s", code)
                self._logger.debug(ast.dump(node))
                return None

    def _tokenize_method(self, method: str) -> str:
        # Add spaces before uppercase letters
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1 \2", method)
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1 \2", s1)

        # Split by period, underscore, and space
        tokens = re.split(r"[._\s]", s2)

        # Remove library alias as we dont want to learn from it
        for library_alias in self._state_stack.imported_names.keys():
            if (
                library_alias in tokens
                and library_alias not in self._state_stack.import_from_modules
            ):
                tokens.remove(library_alias)

        return " ".join(tokens).lower()

    def _tokenize_literal(self, literal: str) -> str:
        # Remove common string delimiters and special characters
        clean_literal = re.sub(r"['\"\[\]\(\)\{\}]", "", literal)
        
        # Add spaces before uppercase letters (CamelCase to space)
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1 \2", clean_literal)
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1 \2", s1)

        # Split by common separators: period, underscore, space, hyphen, and slash
        tokens = re.split(r"[._\s\-/]", s2)

        # Filter out empty strings and convert to lowercase
        return " ".join(filter(None, tokens)).lower()


    def _get_last_variable_version(
        self, variable: str, max_depth: int = 99
    ) -> Optional[str]:
        if variable in self._current_state.variable_versions:
            return self._current_state.variable_versions.get(variable)[-1]

        if self._current_state.parent_context and max_depth > 0:
            current_context = self._current_state.context
            self._state_stack.restore_parent_state()
            self._current_state = self._state_stack.get_current_state()
            variable_version = self._get_last_variable_version(
                variable, max_depth=max_depth - 1
            )
            self._state_stack.restore_state(current_context)
            self._current_state = self._state_stack.get_current_state()
            return variable_version
        else:
            return None

    def _import_accessible(self, name: str, max_depth: int = 99) -> bool:
        if (
            name in self._state_stack.import_from_modules
            or name in self._state_stack.imported_names
        ):
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

    def _get_lr_values(
        self, left: ast.AST, right: ast.AST
    ) -> tuple[Optional[str], Optional[str]]:
        def get_name(node: ast.AST) -> Optional[str]:
            names = self._get_names(node)
            if not names:
                return None

            flat_names = flatten_list(names)
            return flat_names[0] if flat_names else None

        return get_name(left), get_name(right)

    def _get_target_components(self, raw_target_list: list) -> list[list[str]]:
        """
        Takes the raw output of _get_names and returns a clean list of component lists.
        e.g., [['a'], ['b', 'c']] for (a, b.c)
        """
        components = []
        if not raw_target_list:
            return []

        if isinstance(raw_target_list[0], list):  # Nested structure from a tuple
            for sub_list in raw_target_list:
                components.extend(self._get_target_components(sub_list))
        else:  # Base case: a single target's components
            components.append(raw_target_list)
        return components

    def _handle_iterable(self, iterable_node: ast.AST) -> str:
        """
        Handles the processing of an iterable node in a loop or comprehension.
        Creates a temporary node for the iterable, visits it to connect its data flow,
        and returns the name of the temporary iterable node.
        """
        # Global uniqueness prefix
        cell_context = getattr(self, "current_cell_id", "unknown")
        short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"
        
        # Inject short_cell
        temp_iterable_node = f"iterable_{short_cell}_{iterable_node.lineno}_{iterable_node.col_offset}"
        self._add_node(temp_iterable_node, NODE_TYPES["INTERMEDIATE"])

        original_variable = self._current_state.current_variable
        self._current_state.set_current_variable(temp_iterable_node)
        self.visit(iterable_node)
        self._current_state.set_current_variable(original_variable)

        return temp_iterable_node

    def _add_node(self, node_name: str, node_type: int, code: str = "") -> None:
        cell_context = getattr(self, "current_cell_id", "unknown_cell")
        self._current_state.add_node(node_name, node_type, code=code, cell_id=cell_context)

    def _add_edge(
        self, 
        source: str, 
        target: str, 
        label: str,
        edge_type: int, 
        raw_code: str = "",
        lineno: int = -1, 
        col_offset: int = -1, 
        end_lineno: int = -1, 
        end_col_offset: int = -1
    ) -> None:
        cell_context = getattr(self, "current_cell_id", "unknown_cell")
        actual_raw_code = raw_code if raw_code is not None else label

        self._current_state.add_edge(
            source=source, 
            target=target, 
            label=label, 
            edge_type=edge_type, 
            raw_code=actual_raw_code, 
            lineno=lineno, 
            col_offset=col_offset, 
            end_lineno=end_lineno, 
            end_col_offset=end_col_offset, 
            cell_id=cell_context
        )