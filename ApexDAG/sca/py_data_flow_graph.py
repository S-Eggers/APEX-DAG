import ast
import networkx as nx
from logging import Logger
from typing import Optional

from ApexDAG.util.draw import Draw
from ApexDAG.state import Stack, State
from ApexDAG.util.logging import setup_logging

from ApexDAG.sca.constants import NODE_TYPES, EDGE_TYPES, VERBOSE
from ApexDAG.sca.ast_graph import ASTGraph
from ApexDAG.sca.inliner import CallInliner
from ApexDAG.sca.legacy_io import LegacyIOMixin

from ApexDAG.sca.graph_utils import (
    convert_multidigraph_to_digraph,
    get_subgraph,
    save_graph,
    load_graph,
)

from ApexDAG.sca.ast_utils import (
    tokenize_method, 
    tokenize_literal, 
    get_names, 
    get_target_components, 
    get_lr_values,
    get_operator_description,
    flatten_list,
    get_base_name,
    process_arguments
)

class PythonDataFlowGraph(ASTGraph, LegacyIOMixin, ast.NodeVisitor):
    def __init__(self, notebook_path: str = "", replace_dataflow: bool = False) -> None:
        super().__init__()
        self._replace_dataflow = replace_dataflow
        self._logger: Logger = setup_logging(
            f"py_data_flow_graph {notebook_path}", VERBOSE
        )
        self._state_stack: Stack = Stack()
        self._current_state: State = self._state_stack.get_current_state()

    def get_graph(self) -> nx.Graph:
        """Returns the final graph for downstream orchestrators to handle."""
        return self._current_state.get_graph()

    def get_state(self) -> State:
        """Returns the raw state object for direct optimization and filtering."""
        return self._current_state

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            self._state_stack.imported_names[alias.asname or alias.name] = alias.name
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        module = node.module
        for alias in node.names:
            self._state_stack.imported_names[alias.asname or alias.name] = module
            self._state_stack.import_from_modules[alias.asname or alias.name] = module
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        value = node.value
        assignment_code = ast.get_source_segment(self.current_cell_source, node) or ""

        for target in node.targets:
            target_names = get_names(target)
            if not target_names:
                target_names = []

            target_names = flatten_list(target_names)
            for target_name_ in target_names:
                is_obj_attribute = isinstance(target_name_, list)
                target_name = target_name_[0] if is_obj_attribute else target_name_

                self._current_state.set_current_variable(
                    self._get_versioned_name(target_name, node.lineno)
                )
                self._current_state.set_current_target(target_name)

                if isinstance(value, ast.Lambda):
                    base_name = get_base_name(target)
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
                        NODE_TYPES["INTERMEDIATE"], 
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

                if target_name not in self._current_state.variable_versions:
                    self._current_state.variable_versions[target_name] = []
                    previous_version = None
                else:
                    previous_version = self._state_stack.get_last_variable_version(target_name)

                self._current_state.variable_versions[target_name].append(
                    self._current_state.current_variable
                )

                if previous_version and not self._current_state.has_edge(
                    previous_version, self._current_state.current_variable
                ):
                    self._add_edge(
                        source=previous_version,
                        target=self._current_state.current_variable,
                        label="reassign",
                        edge_type=EDGE_TYPES["OMITTED"],
                        raw_code=assignment_code,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        end_lineno=node.end_lineno,
                        end_col_offset=node.end_col_offset,
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

        target_base_name = get_base_name(target)
        new_target_version = self._get_versioned_name(target_base_name, node.lineno)
        self._add_node(new_target_version, NODE_TYPES["VARIABLE"], code=aug_code)

        old_target_version = self._state_stack.get_last_variable_version(target_base_name)
        if old_target_version:
            self._add_edge(
                source=old_target_version,
                target=new_target_version,
                label=operator,
                edge_type=EDGE_TYPES["CALLER"],
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

        if target_base_name not in self._current_state.variable_versions:
            self._current_state.variable_versions[target_base_name] = []
        self._current_state.variable_versions[target_base_name].append(
            new_target_version
        )

        self._current_state.set_current_variable(None)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
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
            function_name = get_names(function_def_node)[0]

            cell_context = getattr(self, "current_cell_id", "unknown")
            short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"
            
            return_node_name = f"return_{function_name}_{short_cell}_{node.lineno}"
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
        base_name = get_base_name(node.value)
        is_call = isinstance(node.value, ast.Call)
        is_first_order = is_call and isinstance(node.value.func, ast.Name)
        is_self_defined = base_name in self._state_stack.functions
        is_imported = (
            base_name in self._state_stack.import_from_modules
            or base_name in self._state_stack.imported_names
        )

        if base_name and (not is_first_order or is_self_defined or is_imported):
            new_version = self._get_versioned_name(base_name, node.lineno)
            
            self._add_node(
                new_version, 
                NODE_TYPES["VARIABLE"],
                code=expr_code
            )
            
            self._current_state.set_current_variable(new_version)
            self._current_state.set_current_target(base_name)
            
            node.value.parent = node
            self.visit(node.value)

            if base_name not in self._current_state.variable_versions:
                self._current_state.variable_versions[base_name] = []

            self._current_state.variable_versions[base_name].append(new_version)

        else:
            cell_context = getattr(self, "current_cell_id", "unknown")
            short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"
            sink_node_name = f"sink_{short_cell}_{node.lineno}_{node.col_offset}"
            
            self._add_node(
                sink_node_name, 
                NODE_TYPES["INTERMEDIATE"], 
                code=expr_code
            )
            
            self._current_state.set_current_variable(sink_node_name)
            node.value.parent = node
            self.visit(node.value)

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
            name = get_names(node)[0]
            name = self._get_versioned_name(name, node.lineno)
            parent_name = get_names(node.parent.targets[0])[0]
            context_name = f"{parent_name}_{name}"
            if context_name in self._state_stack:
                return node

            self._state_stack.functions[parent_name]["context"] = context_name
            self._state_stack.functions[parent_name]["args"] = process_arguments(node.args)
            self._state_stack.functions[parent_name]["is_recursive"] = False
            parent_context = self._current_state.context
            
            with self._state_stack.scope(context_name, parent_context) as scoped_state:
                self._current_state = scoped_state
                cell_context = getattr(self, "current_cell_id", "unknown")
                short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"

                for arg in self._state_stack.functions[parent_name]["args"]["args"]:
                    argument_node = f"{arg}_{short_cell}_{node.lineno}"
                    self._add_node(argument_node, NODE_TYPES["VARIABLE"])
                    self._current_state.variable_versions[arg] = [argument_node]

                self.visit(node.body)
                
            self._current_state = self._state_stack.get_current_state()

        else:
            super().generic_visit(node)
            arguments = process_arguments(node.args)
            for arg in arguments["args"]:
                argument_node = f"{arg}_{node.lineno}"
                self._add_node(argument_node, NODE_TYPES["VARIABLE"])
                self._current_state.variable_versions[arg] = [argument_node]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        function_name = get_names(node)[0]
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
        self._state_stack.functions[function_name]["args"] = process_arguments(node.args)
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
        caller_object_name = None
        function_name = None

        if isinstance(node.func, ast.Attribute):
            caller_object_name = self._get_caller_object(node.func.value)
            function_name = node.func.attr
            
            if isinstance(node.func.value, ast.Call):
                node.func.value.parent = node.func
                
                cell_context = getattr(self, "current_cell_id", "unknown")
                short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"
                intermediate_node = f"chain_{short_cell}_{node.lineno}_{node.func.value.col_offset}"
                
                self._add_node(intermediate_node, NODE_TYPES["INTERMEDIATE"])
                
                original_var = self._current_state.current_variable
                self._current_state.set_current_variable(intermediate_node)
                
                self.visit(node.func.value)
                
                self._current_state.set_current_variable(original_var)
                self._current_state.set_last_variable(intermediate_node)
                caller_object_name = intermediate_node
                
            elif hasattr(node.func, "value"):
                node.func.value.parent = node.func

        elif isinstance(node.func, ast.Name):
            caller_object_name = self._get_caller_object(node.func)
            function_name = node.func.id
        elif isinstance(node.func, (ast.Call, ast.Subscript)):
            caller_object_name = self._get_caller_object(node.func)
            function_name = "__call__"
            self.visit(node.func) 
        else:
            raise NotImplementedError(
                f"Unsupported function call {ast.get_source_segment(self.current_cell_source, node)} with node {ast.dump(node)}"
            )

        if (
            caller_object_name in self._state_stack.imported_names
            or caller_object_name in self._state_stack.import_from_modules
        ):
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self._process_library_call(node, caller_object_name, function_name)
        elif (
            caller_object_name in self._state_stack.classes
            or caller_object_name in self._state_stack.instances
            or function_name in self._state_stack.classes
        ):
            is_instance = caller_object_name in self._state_stack.instances
            self._process_class_call(node, caller_object_name, function_name, is_instance)
        elif (
            not caller_object_name or caller_object_name == function_name
        ) and function_name in self._state_stack.functions:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self._process_function_call(node, function_name)
        elif function_name in [
            "enumerate", "zip", "next", "iter", "range", "sorted", "map", "filter"
        ]:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self._process_builtin_call(node, function_name)
        elif self._current_state.current_target and not (caller_object_name in self._state_stack.instances):
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self._process_method_call(node, caller_object_name, function_name)
        else:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)

        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        left_var, right_var = get_lr_values(node.left, node.right)

        node.left.parent = node
        node.right.parent = node

        self.visit(node.left)
        self.visit(node.right)

        try:
            operator = node.op.__class__.__name__.lower()
        except AttributeError:
            operator = None

        if self._current_state.current_variable and operator:
            raw_code = ast.get_source_segment(self.current_cell_source, node) or operator
            if left_var:
                left_version = self._state_stack.get_last_variable_version(left_var)
                self._add_edge(
                    source=left_version,
                    target=self._current_state.current_variable,
                    label=operator,
                    edge_type=EDGE_TYPES["CALLER"],
                    raw_code=raw_code,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=node.end_lineno,
                    end_col_offset=node.end_col_offset,
                )
            if right_var:
                right_version = self._state_stack.get_last_variable_version(right_var)
                self._add_edge(
                    source=right_version,
                    target=self._current_state.current_variable,
                    label=operator,
                    edge_type=EDGE_TYPES["CALLER"],
                    raw_code=raw_code,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=node.end_lineno,
                    end_col_offset=node.end_col_offset,
                )

        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        node.left.parent = node
        self.visit(node.left)
        left_operand = node.left
        raw_code = ast.get_source_segment(self.current_cell_source, node) or ""

        for i, op in enumerate(node.ops):
            right_operand = node.comparators[i]
            right_operand.parent = node
            self.visit(right_operand)

            left_var, right_var = get_lr_values(left_operand, right_operand)
            operator = get_operator_description(op)

            if self._current_state.current_variable and operator:
                if left_var:
                    left_version = self._state_stack.get_last_variable_version(left_var)
                    self._add_edge(
                        source=left_version,
                        target=self._current_state.current_variable,
                        label=operator,
                        edge_type=EDGE_TYPES["CALLER"],
                        raw_code=raw_code,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        end_lineno=node.end_lineno,
                        end_col_offset=node.end_col_offset,
                    )
                if right_var:
                    right_version = self._state_stack.get_last_variable_version(right_var)
                    self._add_edge(
                        source=right_version,
                        target=self._current_state.current_variable,
                        label=operator,
                        edge_type=EDGE_TYPES["CALLER"],
                        raw_code=raw_code,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        end_lineno=node.end_lineno,
                        end_col_offset=node.end_col_offset,
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
            self._process_library_attr(node, var_name)
            return node
        else:
            var_version = self._state_stack.get_last_variable_version(var_name)
            is_base_of_attribute = hasattr(node, "parent") and isinstance(
                node.parent, ast.Attribute
            )
            if is_base_of_attribute:
                self._current_state.set_last_variable(var_version)
                return node

            edge_type = EDGE_TYPES["INPUT"]
            if hasattr(node, "parent") and isinstance(node.parent, ast.Subscript):
                code_segment, edge_type = self._process_subscript(node)
            else:
                code_segment = var_name

            label = tokenize_method(code_segment, self._state_stack.imported_names, self._state_stack.import_from_modules)

            if self._current_state.current_variable:
                self._add_edge(
                    source=var_version,
                    target=self._current_state.current_variable,
                    label=label,
                    edge_type=edge_type,
                    raw_code=code_segment,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=node.end_lineno,
                    end_col_offset=node.end_col_offset,
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

        label = tokenize_method(node.attr, self._state_stack.imported_names, self._state_stack.import_from_modules)

        self._add_edge(
            source=var_name,
            target=self._current_state.current_variable,
            label=label,
            edge_type=EDGE_TYPES["CALLER"],
            raw_code=node.attr,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
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

            tokenized_label = tokenize_literal(truncated_literal)
            
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
            source=self._state_stack.get_last_variable_version(self._current_state.current_target),
            target=self._current_state.current_variable,
            label="if",
            edge_type=EDGE_TYPES["CALLER"],
            raw_code="if",
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
        )
        self.visit(node.body)
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
            with self._state_stack.scope(if_context, parent_context) as scoped_state:
                self._current_state = scoped_state
                for stmt in node.body:
                    stmt.parent = node
                    stmt.parent_context = if_context
                    self.visit(stmt)
                branch_state = self._current_state
            self._state_stack.branches.append((branch_state, "else if", EDGE_TYPES["BRANCH"]))
            if_branch = False
            previous_target = None
        else:
            node_name = get_names(node)[0]
            var_version = self._get_versioned_name(node_name, node.lineno)
            previous_target = self._current_state.current_target
            self._current_state.current_target = var_version

            if_context = f"{var_version}_if"
            with self._state_stack.scope(if_context, parent_context) as scoped_state:
                self._current_state = scoped_state
                for stmt in node.body:
                    stmt.parent = node
                    stmt.parent_context = if_context
                    self.visit(stmt)
                branch_state = self._current_state
            self._state_stack.branches.append((branch_state, "if", EDGE_TYPES["BRANCH"]))

        self._current_state = self._state_stack.get_current_state()

        if node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node.orelse[0].parent = node
            node.orelse[0].parent_context = node.parent_context
            self.visit(node.orelse[0])
        elif len(node.orelse) > 0:
            cell_context = getattr(self, "current_cell_id", "unknown_cell")
            else_context = f"else_{cell_context}_{node.lineno}"
            with self._state_stack.scope(else_context, parent_context) as scoped_state:
                self._current_state = scoped_state
                for stmt in node.orelse:
                    stmt.parent = node
                    stmt.parent_context = else_context
                    self.visit(stmt)
                branch_state = self._current_state
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
        node_name = get_names(node)[0]
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
        raw_targets = get_names(node.target)
        all_targets = get_target_components(raw_targets) if raw_targets else []

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
                    source=temp_iterable_node,
                    target=target_version,
                    label=edge_label,
                    edge_type=EDGE_TYPES["LOOP"],
                    raw_code=edge_label,
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
        
        cell_context = getattr(self, "current_cell_id", "unknown")
        short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"

        final_list_variable = self._current_state.current_variable
        if not final_list_variable:
            final_list_variable = f"list_comp_result_{short_cell}_{node.lineno}"
            self._add_node(final_list_variable, NODE_TYPES["VARIABLE"])

        parent_context = self._current_state.context
        if self._state_stack.nested:
            list_comp_context = f"{parent_context}_nested_list_comp_{short_cell}_{node.lineno}"
        else:
            list_comp_context = f"list_comp_{short_cell}_{node.lineno}"
            
        with self._state_stack.scope(list_comp_context, parent_context) as scoped_state:
            self._current_state = scoped_state

            start_loop_node = f"{final_list_variable}_start_loop"
            self._add_node(start_loop_node, NODE_TYPES["LOOP"])
            
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
                raw_targets = get_names(generator.target)
                all_targets = get_target_components(raw_targets) if raw_targets else []
                
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
                raw_targets = get_names(generator.target)
                all_targets = (
                    get_target_components(raw_targets) if raw_targets else []
                )
                for components in all_targets:
                    base_name = components[0]
                    if base_name in self._current_state.variable_versions:
                        del self._current_state.variable_versions[base_name]

        self._current_state = self._state_stack.get_current_state()
        
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

    def _check_resursion(self, node: ast.FunctionDef) -> bool:
        function_name = node.name
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                if sub_node.func.id == function_name:
                    return True
        return False

    def _process_method_call(
        self, node: ast.Call, caller_object_name: str, tokens: Optional[str]
    ) -> None:
        previous_version = self._state_stack.get_last_variable_version(caller_object_name)
        previous_version = (
            previous_version if previous_version else self._current_state.last_variable
        )
        label = tokenize_method(tokens, self._state_stack.imported_names, self._state_stack.import_from_modules)
        raw_code = ast.get_source_segment(self.current_cell_source, node) or tokens
        
        self._add_edge(
            source=previous_version,
            target=self._current_state.current_variable,
            label=label,
            edge_type=EDGE_TYPES["CALLER"],
            raw_code=raw_code,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
        )

        for arg in node.args:
            if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                self._process_name_attr_sub_args(node, arg)
            elif isinstance(arg, (ast.Tuple)):
                arg_names = get_names(arg)
                if not arg_names:
                    continue

                processed_arg_names = [
                    flatten_list(name_list)[0]
                    for name_list in arg_names
                    if name_list and flatten_list(name_list)
                ]

                for arg_name in processed_arg_names:
                    if arg_name:
                        arg_version = self._state_stack.get_last_variable_version(arg_name)
                        code_segment = tokenize_method(arg_name, self._state_stack.imported_names, self._state_stack.import_from_modules)
                        self._add_edge(
                            source=arg_version,
                            target=self._current_state.current_variable,
                            label=code_segment,
                            edge_type=EDGE_TYPES["INPUT"],
                            raw_code=str(arg_name),
                            lineno=node.lineno,
                            col_offset=node.col_offset,
                            end_lineno=node.end_lineno,
                            end_col_offset=node.end_col_offset,
                        )

    def _process_library_call(
        self, node: ast.Call, caller_object_name: str, tokens: str = None
    ) -> None:
        import_node = self._state_stack.imported_names[caller_object_name]
        self._add_node(import_node, NODE_TYPES["IMPORT"])
        
        label_raw = tokens if tokens else ast.get_source_segment(self.current_cell_source, node.func)
        label = tokenize_method(label_raw, self._state_stack.imported_names, self._state_stack.import_from_modules)
        raw_code = ast.get_source_segment(self.current_cell_source, node) or label_raw
        
        self._add_edge(
            source=import_node,
            target=self._current_state.current_variable,
            label=label,
            edge_type=EDGE_TYPES["FUNCTION_CALL"],
            raw_code=raw_code,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
        )

        for arg in node.args:
            if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                self._process_name_attr_sub_args(node, arg)

        self._current_state.set_last_variable(import_node)

    def _process_builtin_call(self, node: ast.Call, function_name: str) -> None:
        if not self._current_state.current_variable:
            return

        builtin_node = "__builtins__"
        self._add_node(builtin_node, NODE_TYPES["IMPORT"])

        # FIX: Extract the actual source snippet for the Edge Sidebar
        raw_code = ast.get_source_segment(self.current_cell_source, node) or function_name

        self._add_edge(
            source=builtin_node,
            target=self._current_state.current_variable,
            label=function_name,
            edge_type=EDGE_TYPES["CALLER"],
            raw_code=raw_code,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
        )

        for arg in node.args:
            arg_names = get_names(arg)
            if arg_names:
                base_name = flatten_list(arg_names)[0]
                if base_name and isinstance(base_name, str):
                    arg_version = self._state_stack.get_last_variable_version(base_name)
                    if arg_version:
                        self._add_edge(
                            source=arg_version,
                            target=self._current_state.current_variable,
                            label="arg",
                            edge_type=EDGE_TYPES["INPUT"],
                            raw_code=str(base_name),
                            lineno=arg.lineno,
                            col_offset=arg.col_offset,
                            end_lineno=arg.end_lineno,
                            end_col_offset=arg.end_col_offset,
                        )

    def _process_name_attr_sub_args(self, node: ast.Call, arg: ast.AST) -> None:
        arg_names = get_names(arg)
        if not arg_names:
            return
        arg_name = flatten_list(arg_names)[0]

        if arg_name:
            arg_version = self._state_stack.get_last_variable_version(arg_name)
            code_segment = tokenize_method(arg_name, self._state_stack.imported_names, self._state_stack.import_from_modules)
            self._add_edge(
                source=arg_version,
                target=self._current_state.current_variable,
                label=code_segment,
                edge_type=EDGE_TYPES["INPUT"],
                raw_code=str(arg_name),
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
            )

    def _process_class_call(
        self, node: ast.Call, caller_object_name: str, tokens: str = None, is_instance: bool = False
    ) -> None:
        inliner = CallInliner(self)
        inliner.process_class(node, caller_object_name, tokens, is_instance)

    def _process_function_call(self, node: ast.Call, tokens: str = None) -> None:
        inliner = CallInliner(self)
        inliner.process_function(node, tokens)

    def _process_library_attr(
        self, node: ast.Attribute, caller_object_name: str
    ) -> None:
        import_node = self._state_stack.imported_names[caller_object_name]
        self._add_node(import_node, NODE_TYPES["IMPORT"])
        
        raw_code = ast.get_source_segment(self.current_cell_source, node) or ""
        label = tokenize_method(raw_code, self._state_stack.imported_names, self._state_stack.import_from_modules)
        
        self._add_edge(
            source=import_node,
            target=self._current_state.current_variable,
            label=label,
            edge_type=EDGE_TYPES["CALLER"],
            raw_code=raw_code,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
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
            code = ast.get_source_segment(self.current_cell_source, node)
            code_segment = ast.get_source_segment(self.current_cell_source, node.parent)
            edge_type = EDGE_TYPES["INPUT"]

        return code_segment, edge_type

    def _get_caller_object(self, value: ast.AST) -> str:
        names = get_names(value)
        if names and (
            self._state_stack.import_accessible(names[0])
            or self._state_stack.class_accessible(names[0])
            or self._state_stack.function_accessible(names[0])
            or names[0] in self._current_state.variable_versions
        ):
            return names[0]

        return self._current_state.current_target

    def _get_versioned_name(self, var_name: str, lineno: int) -> str:
        cell_context = getattr(self, "current_cell_id", "unknown")
        short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"
        
        return f"{var_name}_{short_cell}_{lineno}"

    def _handle_iterable(self, iterable_node: ast.AST) -> str:
        cell_context = getattr(self, "current_cell_id", "unknown")
        short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"
        
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