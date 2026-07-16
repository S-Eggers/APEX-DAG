import ast
import logging
import re

import networkx as nx

from SystemX.sca.ast_graph import ASTGraph
from SystemX.sca.ast_utils import (
    flatten_list,
    get_base_name,
    get_lr_values,
    get_names,
    get_operator_description,
    get_target_components,
    process_arguments,
    tokenize_literal,
    tokenize_method,
)
from SystemX.sca.constants import EDGE_TYPES, NODE_TYPES
from SystemX.sca.inliner import CallInliner
from SystemX.sca.legacy_io import LegacyIOMixin
from SystemX.state import Stack, State

from .types.dsl_policy import DslPolicy, NoDslPolicy
from .types.inlining_policy import InliningPolicy, NoInliningPolicy
from .types.provenance_policy import NoProvenancePolicy, ProvenancePolicy

logger = logging.getLogger(__name__)

class CDFIntermediateRepresentation(ASTGraph, LegacyIOMixin, ast.NodeVisitor):
    def __init__(
        self,
        notebook_path: str = "",
        inlining_policy: InliningPolicy | None = None,
        provenance_policy: ProvenancePolicy | None = None,
        dsl_policy: DslPolicy | None = None,
    ) -> None:
        super().__init__()
        self._state_stack: Stack = Stack()
        self._current_state: State = self._state_stack.get_current_state()
        self._scope_counter = 0
        self._visited_ast_nodes: set[int] = set()
        self._policy = inlining_policy or NoInliningPolicy()
        self._provenance_policy = provenance_policy or NoProvenancePolicy()
        self._inliner = CallInliner(context=self, policy=self._policy)
        self._dsl_session = (dsl_policy or NoDslPolicy()).new_session()

    def pre_parse(self, sources: list[str]) -> None:
        self._dsl_session.pre_scan(sources)

    def transform_tree(self, tree: ast.Module) -> ast.Module:
        return self._dsl_session.rewrite(tree)

    @property
    def current_state(self) -> State:
        return self._current_state

    @current_state.setter
    def current_state(self, value: State) -> None:
        self._current_state = value

    @property
    def state_stack(self) -> Stack:
        return self._state_stack

    def add_node(self, node_name: str, node_type: int, code: str = "") -> None:
        self._add_node(node_name, node_type, code)

    def add_edge(self, source: str, target: str, label: str, edge_type: int, raw_code: str = "", lineno: int = -1, col_offset: int = -1, end_lineno: int = -1, end_col_offset: int = -1) -> None:
        self._add_edge(source, target, label, edge_type, raw_code, lineno, col_offset, end_lineno, end_col_offset)

    def get_graph(self) -> nx.Graph:
        """Returns the final graph for downstream orchestrators to handle."""
        return self._current_state.get_graph()

    def enrich_provenance(self) -> "CDFIntermediateRepresentation":
        """Explicitly triggers in-place mutation of the underlying graph structure to calculate transitive dependencies."""
        underlying_graph = self.get_graph()
        self._provenance_policy.apply(underlying_graph)
        return self

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

                self._current_state.set_current_variable(self._get_versioned_name(target_name, node.lineno))
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
                        code=assignment_code,
                    )
                else:
                    self._add_node(
                        self._current_state.current_variable,
                        NODE_TYPES["VARIABLE"],
                        code=assignment_code,
                    )

                value.parent = node
                self.visit(value)

                if target_name not in self._current_state.variable_versions:
                    self._current_state.variable_versions[target_name] = []
                    previous_version = None
                else:
                    previous_version = self._state_stack.get_last_variable_version(target_name)

                self._current_state.variable_versions[target_name].append(self._current_state.current_variable)

                if previous_version:
                    has_incoming_connections = self._current_state._G.in_degree(self._current_state.current_variable) > 0
                    if not has_incoming_connections:
                        self._add_edge(
                            source=previous_version,
                            target=self._current_state.current_variable,
                            label="reassign",
                            edge_type=EDGE_TYPES["OMITTED"],
                            raw_code=assignment_code,
                            lineno=node.lineno,
                            col_offset=node.col_offset,
                            end_lineno=getattr(node, "end_lineno", -1),
                            end_col_offset=getattr(node, "end_col_offset", -1),
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

        cell_context = getattr(self, "current_cell_id", "unk")
        short_cell = str(cell_context)[:8]
        op_node_name = f"op_aug_{operator}_{short_cell}_{node.lineno}_{node.col_offset}"
        self._add_node(op_node_name, NODE_TYPES["CALL"], code=aug_code, label=operator.replace("_", " "))

        if old_target_version:
            self._add_edge(
                source=old_target_version,
                target=op_node_name,
                label=operator.replace("_", " "),
                edge_type=EDGE_TYPES["CALLER"],
                raw_code=aug_code,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
            )

        original_current_var = self._current_state.current_variable
        self._current_state.set_current_variable(op_node_name)
        self.visit(value)
        self._current_state.set_current_variable(original_current_var)

        self._add_edge(
            source=op_node_name,
            target=new_target_version,
            label="output",
            edge_type=EDGE_TYPES["OMITTED"],
            raw_code=aug_code,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
        )

        if target_base_name not in self._current_state.variable_versions:
            self._current_state.variable_versions[target_base_name] = []
        self._current_state.variable_versions[target_base_name].append(new_target_version)

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
                self._state_stack.functions[function_name]["return_nodes"].append(return_node_name)

            original_variable = self._current_state.current_variable
            self._current_state.set_current_variable(return_node_name)
            self.visit(node.value)
            self._current_state.set_current_variable(original_variable)

        return node

    def visit_Expr(self, node: ast.Expr) -> None:
        expr_code = ast.get_source_segment(self.current_cell_source, node) or ""
        base_name = get_base_name(node.value)
        is_call = isinstance(node.value, ast.Call)
        is_first_order = is_call and isinstance(node.value.func, ast.Name)
        is_self_defined = base_name in self._state_stack.functions
        is_imported = base_name in self._state_stack.import_from_modules or base_name in self._state_stack.imported_names
        is_direct_method_on_var = is_call and not is_first_order and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
        is_potential_mutation = is_call and base_name and not is_imported and (is_direct_method_on_var or is_self_defined)

        if is_potential_mutation:
            new_version = self._get_versioned_name(base_name, node.lineno)

            self._add_node(new_version, NODE_TYPES["VARIABLE"], code=expr_code)

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

            self._add_node(sink_node_name, NODE_TYPES["INTERMEDIATE"], code=expr_code)

            self._current_state.set_current_variable(sink_node_name)
            node.value.parent = node
            self.visit(node.value)

        self._current_state.set_current_variable(None)
        self._current_state.set_current_target(None)
        self._current_state.set_last_variable(None)
        self._current_state.edge_for_current_target = {}

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        if id(node) in self._visited_ast_nodes:
            return node
        self._visited_ast_nodes.add(id(node))

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
        self._visited_ast_nodes.add(id(node))

        function_name = get_names(node)[0]
        node_name = self._get_versioned_name(function_name, node.lineno)
        context_name = self._generate_context_name(f"func_{function_name}", node)

        self._state_stack.functions[function_name] = {
            "node": node_name,
            "context": context_name,
            "is_recursive": self._check_recursion(node),
            "kwargs": bool(node.args.kwarg),
            "vararg": bool(node.args.vararg),
            "return_nodes": [],
            "parent_class": node.parent.name if isinstance(getattr(node, "parent", None), ast.ClassDef) else None,
        }
        self._state_stack.functions[function_name]["args"] = process_arguments(node.args)

        self._add_node(node_name, NODE_TYPES["FUNCTION"], label=f"declare {function_name.replace('_', ' ')}")

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

            if not isinstance(node.func.value, ast.Name):
                node.func.value.parent = node.func
                cell_context = getattr(self, "current_cell_id", "unknown")
                short_cell = str(cell_context)[:8]

                intermediate_node = f"eval_base_{short_cell}_{node.lineno}_{node.end_col_offset}"
                self._add_node(intermediate_node, NODE_TYPES["INTERMEDIATE"])

                original_var = self._current_state.current_variable
                self._current_state.set_current_variable(intermediate_node)

                self.visit(node.func.value)

                self._current_state.set_current_variable(original_var)
                self._current_state.set_last_variable(intermediate_node)
                caller_object_name = intermediate_node
            else:
                caller_object_name = self._get_caller_object(node.func.value)
                node.func.value.parent = node.func

        elif isinstance(node.func, ast.Name):
            caller_object_name = self._get_caller_object(node.func)
            function_name = node.func.id
        elif isinstance(node.func, (ast.Call, ast.Subscript)):
            caller_object_name = self._get_caller_object(node.func)
            function_name = "__call__"
            self.visit(node.func)
        else:
            raise NotImplementedError(f"Unsupported call: {ast.dump(node)}")

        if caller_object_name in self._state_stack.imported_names or caller_object_name in self._state_stack.import_from_modules:
            self._process_library_call(node, caller_object_name, function_name)

        elif caller_object_name in self._state_stack.classes or caller_object_name in self._state_stack.instances or function_name in self._state_stack.classes:
            is_instance = caller_object_name in self._state_stack.instances
            self._process_class_call(node, caller_object_name, function_name, is_instance)

        elif (not caller_object_name or caller_object_name == function_name) and function_name in self._state_stack.functions:
            self._process_function_call(node, function_name)

        elif function_name in ["enumerate", "zip", "next", "iter", "range", "sorted", "map", "filter"]:
            self._process_builtin_call(node, function_name)

        elif self._current_state.current_target and caller_object_name not in self._state_stack.instances:
            self._process_method_call(node, caller_object_name, function_name)

        else:
            self._process_generic_call(node, caller_object_name, function_name)

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

            cell_context = getattr(self, "current_cell_id", "unk")
            op_node_name = f"op_{operator}_{str(cell_context)[:8]}_{node.lineno}_{node.col_offset}"
            self._add_node(op_node_name, NODE_TYPES["CALL"], code=raw_code, label=operator.replace("_", " "))

            if left_var:
                left_version = self._state_stack.get_last_variable_version(left_var)
                self._add_edge(
                    source=left_version,
                    target=op_node_name,
                    label="left operand",
                    edge_type=EDGE_TYPES["INPUT"],
                    raw_code=raw_code,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
            if right_var:
                right_version = self._state_stack.get_last_variable_version(right_var)
                self._add_edge(
                    source=right_version,
                    target=op_node_name,
                    label="right operand",
                    edge_type=EDGE_TYPES["INPUT"],
                    raw_code=raw_code,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )

            self._add_edge(
                source=op_node_name,
                target=self._current_state.current_variable,
                label="output",
                edge_type=EDGE_TYPES["OMITTED"],
                raw_code=raw_code,
                lineno=node.lineno,
                col_offset=node.col_offset,
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
                cell_context = getattr(self, "current_cell_id", "unk")
                op_node_name = f"op_cmp_{operator}_{str(cell_context)[:8]}_{node.lineno}_{node.col_offset}"
                self._add_node(op_node_name, NODE_TYPES["CALL"], code=raw_code)

                if left_var:
                    left_version = self._state_stack.get_last_variable_version(left_var)
                    self._add_edge(
                        source=left_version,
                        target=op_node_name,
                        label=operator,
                        edge_type=EDGE_TYPES["INPUT"],
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
                        target=op_node_name,
                        label=operator,
                        edge_type=EDGE_TYPES["INPUT"],
                        raw_code=raw_code,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        end_lineno=node.end_lineno,
                        end_col_offset=node.end_col_offset,
                    )

                self._add_edge(
                    source=op_node_name,
                    target=self._current_state.current_variable,
                    label="output",
                    edge_type=EDGE_TYPES["OMITTED"],
                    raw_code=raw_code,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=node.end_lineno,
                    end_col_offset=node.end_col_offset,
                )

            left_operand = right_operand

        return node

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return

        var_name = node.id
        if var_name in self._state_stack.imported_names or var_name in self._state_stack.import_from_modules:
            self._process_library_attr(node, var_name)
            return

        var_version = self._state_stack.get_last_variable_version(var_name)

        edge_type = EDGE_TYPES["INPUT"]
        label = tokenize_method(
            var_name,
            self._state_stack.imported_names,
            self._state_stack.import_from_modules,
        )

        if self._current_state.current_variable:
            self._add_edge(
                source=var_version,
                target=self._current_state.current_variable,
                label=label,
                edge_type=edge_type,
                raw_code=var_name,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=getattr(node, "end_lineno", -1),
                end_col_offset=getattr(node, "end_col_offset", -1),
            )

        self._current_state.set_last_variable(var_version)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        cell_id = getattr(self, "current_cell_id", "unk")

        attr_node_name = f"attr_{node.attr}_{str(cell_id)[:8]}_{node.lineno}_{node.col_offset}"
        raw_code = ast.get_source_segment(self.current_cell_source, node) or node.attr
        self._add_node(attr_node_name, NODE_TYPES["INTERMEDIATE"], code=raw_code)

        original_var = self._current_state.current_variable
        self._current_state.set_current_variable(attr_node_name)

        if hasattr(node, "value"):
            node.value.parent = node
            self.visit(node.value)

        self._current_state.set_current_variable(original_var)

        label = tokenize_method(
            node.attr,
            self._state_stack.imported_names,
            self._state_stack.import_from_modules,
        )

        if original_var:
            self._add_edge(
                source=attr_node_name,
                target=original_var,
                label=label,
                edge_type=EDGE_TYPES["CALLER"],
                raw_code=node.attr,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=getattr(node, "end_lineno", -1),
                end_col_offset=getattr(node, "end_col_offset", -1),
            )

        self._current_state.set_last_variable(attr_node_name)

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
        is_elif = hasattr(node, "parent") and isinstance(node.parent, ast.If) and node.parent.orelse and node.parent.orelse[0] == node
        parent_context = node.parent_context if is_elif else self._current_state.context
        node.parent_context = parent_context

        if is_elif:
            if_context = self._generate_context_name("else_if", node)
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
            if_context = self._generate_context_name("if", node)
            previous_target = self._current_state.current_target
            self._current_state.current_target = if_context

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
            else_context = self._generate_context_name("else", node)
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
        parent_context = self._current_state.context
        cell_id = getattr(self, "current_cell_id", "unk")
        while_context = self._generate_context_name("while", node)

        while_op_node = f"while_op_{str(cell_id)[:8]}_{node.lineno}_{node.col_offset}"
        self._add_node(while_op_node, NODE_TYPES["LOOP"], code="while")

        original_var = self._current_state.current_variable
        self._current_state.set_current_variable(while_op_node)
        self.visit(node.test)
        self._current_state.set_current_variable(original_var)

        with self._state_stack.scope(while_context, parent_context) as scoped_state:
            self._current_state = scoped_state
            for stmt in node.body:
                stmt.parent = node
                self.visit(stmt)
            contexts = [(self._current_state, "start_loop", EDGE_TYPES["LOOP"])]

            modified_vars = list(scoped_state.variable_versions.keys())

        self._current_state = self._state_stack.get_current_state()

        if node.orelse and len(node.orelse) > 0:
            else_context = self._generate_context_name("while_else", node)
            with self._state_stack.scope(else_context, parent_context) as scoped_state:
                self._current_state = scoped_state
                for stmt in node.orelse:
                    stmt.parent = node
                    self.visit(stmt)
                contexts.append((self._current_state, "else", EDGE_TYPES["BRANCH"]))

                for var in scoped_state.variable_versions:
                    if var not in modified_vars:
                        modified_vars.append(var)

            self._current_state = self._state_stack.get_current_state()

        cell_context = getattr(self, "current_cell_id", "unknown_cell")

        self._state_stack.merge_states(parent_context, contexts, cell_id=cell_context, hub_node=while_op_node)
        self._current_state = self._state_stack.get_current_state()

        for var in modified_vars:
            post_loop_version = f"{var}_final_{str(cell_id)[:8]}_{node.lineno}"
            self._add_node(post_loop_version, NODE_TYPES["VARIABLE"])
            self._add_edge(while_op_node, post_loop_version, "loop_exit", EDGE_TYPES["OMITTED"], lineno=node.lineno)

            if var not in self._current_state.variable_versions:
                self._current_state.variable_versions[var] = []
            self._current_state.variable_versions[var].append(post_loop_version)

        return node

    def visit_For(self, node: ast.For) -> ast.For:
        iterable_state_node = self._handle_iterable(node.iter)
        raw_targets = get_names(node.target)
        all_targets = get_target_components(raw_targets) if raw_targets else []

        parent_context = self._current_state.context
        cell_id = getattr(self, "current_cell_id", "unk")
        for_context = self._generate_context_name("for_loop", node)

        loop_op_node = f"for_op_{str(cell_id)[:8]}_{node.lineno}_{node.col_offset}"
        self._add_node(loop_op_node, NODE_TYPES["LOOP"], code="for_loop")

        if iterable_state_node:
            self._add_edge(iterable_state_node, loop_op_node, "iterable", EDGE_TYPES["INPUT"], lineno=node.lineno)

        with self._state_stack.scope(for_context, parent_context) as scoped_state:
            self._current_state = scoped_state
            for components in all_targets:
                base_name = components[0]
                target_version = self._get_versioned_name(base_name, node.lineno)
                self._add_node(target_version, NODE_TYPES["VARIABLE"])
                self._current_state.variable_versions[base_name] = [target_version]

                self._add_edge(loop_op_node, target_version, "iterate", EDGE_TYPES["LOOP"], lineno=node.lineno)

            for stmt in node.body:
                self.visit(stmt)

            contexts = [(self._current_state, "loop_body", EDGE_TYPES["LOOP"])]

        self._current_state = self._state_stack.get_current_state()
        self._state_stack.merge_states(parent_context, contexts, cell_id=cell_id, hub_node=loop_op_node)

        for components in all_targets:
            base_name = components[0]
            post_loop_version = f"{base_name}_final_{str(cell_id)[:8]}_{node.lineno}"
            self._add_node(post_loop_version, NODE_TYPES["VARIABLE"])

            self._add_edge(loop_op_node, post_loop_version, "loop_exit", EDGE_TYPES["OMITTED"], lineno=node.lineno)
            self._current_state.variable_versions[base_name].append(post_loop_version)

        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        raw_code = ast.get_source_segment(self.current_cell_source, node) or "subscript"

        cell_id = getattr(self, "current_cell_id", "unk")
        op_node = f"subscript_{str(cell_id)[:8]}_{node.lineno}_{node.col_offset}"
        self._add_node(op_node, NODE_TYPES["CALL"], code=raw_code)

        original_var = self._current_state.current_variable
        self._current_state.set_current_variable(op_node)

        node.value.parent = node
        self.visit(node.value)

        node.slice.parent = node
        self.visit(node.slice)

        self._current_state.set_current_variable(original_var)
        if original_var:
            self._add_edge(
                source=op_node,
                target=original_var,
                label="output",
                edge_type=EDGE_TYPES["OMITTED"],
                raw_code=raw_code,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=getattr(node, "end_lineno", -1),
                end_col_offset=getattr(node, "end_col_offset", -1),
            )

        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        if id(node) in self._visited_ast_nodes:
            return node
        self._visited_ast_nodes.add(id(node))

        cell_context = getattr(self, "current_cell_id", "unknown")
        short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"

        final_list_variable = self._current_state.current_variable
        if not final_list_variable:
            final_list_variable = f"list_comp_result_{short_cell}_{node.lineno}"
            self._add_node(final_list_variable, NODE_TYPES["VARIABLE"])

        parent_context = self._current_state.context
        list_comp_context = self._generate_context_name("list_comp", node)

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
                    target_version = self._get_versioned_name(base_name, generator.target.lineno)
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
                all_targets = get_target_components(raw_targets) if raw_targets else []
                for components in all_targets:
                    base_name = components[0]
                    if base_name in self._current_state.variable_versions:
                        del self._current_state.variable_versions[base_name]

        self._current_state = self._state_stack.get_current_state()

        cell_context_full = getattr(self, "current_cell_id", "unknown_cell")
        self._state_stack.merge_states(
            parent_context,
            [
                (
                    self._state_stack._state[list_comp_context],
                    "list_comp",
                    EDGE_TYPES["CALLER"],
                )
            ],
            cell_id=cell_context_full,
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

    def _check_recursion(self, node: ast.FunctionDef) -> bool:
        function_name = node.name
        return any(isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name) and sub_node.func.id == function_name for sub_node in ast.walk(node))

    def _process_library_call(self, node: ast.Call, caller_object_name: str, tokens: str | None = None) -> None:
        import_node = self._state_stack.imported_names[caller_object_name]
        self._add_node(import_node, NODE_TYPES["IMPORT"])

        label_raw = tokens if tokens else ast.get_source_segment(self.current_cell_source, node.func)
        label = tokenize_method(label_raw, self._state_stack.imported_names, self._state_stack.import_from_modules)
        raw_code = ast.get_source_segment(self.current_cell_source, node) or label_raw

        cell_id = getattr(self, "current_cell_id", "unk")
        op_node = f"call_{label}_{str(cell_id)[:8]}_{node.lineno}_{node.col_offset}"
        self._add_node(op_node, NODE_TYPES["CALL"], code=raw_code)

        self._add_edge(import_node, op_node, label, EDGE_TYPES["FUNCTION_CALL"], raw_code=raw_code, lineno=node.lineno)

        original_var = self._current_state.current_variable
        self._current_state.set_current_variable(op_node)

        self._visit_call_args(node, op_node)

        self._current_state.set_current_variable(original_var)
        if original_var:
            self._add_edge(op_node, original_var, "output", EDGE_TYPES["OMITTED"], raw_code=raw_code, lineno=node.lineno)

        self._current_state.set_last_variable(op_node)

    def _process_method_call(self, node: ast.Call, caller_object_name: str, tokens: str | None) -> None:
        previous_version = self._state_stack.get_last_variable_version(caller_object_name)
        previous_version = previous_version if previous_version else self._current_state.last_variable
        label = tokenize_method(tokens, self._state_stack.imported_names, self._state_stack.import_from_modules)
        raw_code = ast.get_source_segment(self.current_cell_source, node) or tokens

        cell_id = getattr(self, "current_cell_id", "unk")
        op_node = f"method_{label}_{str(cell_id)[:8]}_{node.lineno}_{node.col_offset}"
        self._add_node(op_node, NODE_TYPES["CALL"], code=raw_code)

        self._add_edge(previous_version, op_node, label, EDGE_TYPES["CALLER"], raw_code=raw_code, lineno=node.lineno)

        original_var = self._current_state.current_variable
        self._current_state.set_current_variable(op_node)

        self._visit_call_args(node, op_node)

        self._current_state.set_current_variable(original_var)
        if original_var:
            self._add_edge(op_node, original_var, "output", EDGE_TYPES["OMITTED"], raw_code=raw_code, lineno=node.lineno)

        self._current_state.set_last_variable(op_node)

    def _process_builtin_call(self, node: ast.Call, function_name: str) -> None:
        builtin_node = "__builtins__"
        self._add_node(builtin_node, NODE_TYPES["IMPORT"])
        raw_code = ast.get_source_segment(self.current_cell_source, node) or function_name

        cell_id = getattr(self, "current_cell_id", "unk")
        op_node = f"builtin_{function_name}_{str(cell_id)[:8]}_{node.lineno}_{node.col_offset}"
        self._add_node(op_node, NODE_TYPES["CALL"], code=raw_code)

        self._add_edge(builtin_node, op_node, function_name, EDGE_TYPES["CALLER"], raw_code=raw_code, lineno=node.lineno)

        original_var = self._current_state.current_variable
        self._current_state.set_current_variable(op_node)

        self._visit_call_args(node, op_node)

        self._current_state.set_current_variable(original_var)
        if original_var:
            self._add_edge(op_node, original_var, "output", EDGE_TYPES["OMITTED"], raw_code=raw_code, lineno=node.lineno)

        self._current_state.set_last_variable(op_node)

    def _process_class_call(
        self,
        node: ast.Call,
        caller_object_name: str,
        tokens: str | None = None,
        is_instance: bool = False,
    ) -> None:
        self._inliner.process_class(node, caller_object_name, tokens or "", is_instance)

    def _process_function_call(self, node: ast.Call, tokens: str | None = None) -> None:
        self._inliner.process_function(node, tokens or "")

    def _process_library_attr(self, node: ast.Attribute, caller_object_name: str) -> None:
        import_node = self._state_stack.imported_names[caller_object_name]
        self._add_node(import_node, NODE_TYPES["IMPORT"])

        raw_code = ast.get_source_segment(self.current_cell_source, node) or ""
        label = tokenize_method(
            raw_code,
            self._state_stack.imported_names,
            self._state_stack.import_from_modules,
        )

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

    def _get_caller_object(self, value: ast.AST) -> str:
        base_name = get_base_name(value)

        if base_name:
            return base_name

        return self._current_state.current_target

    def _get_versioned_name(self, var_name: str, lineno: int) -> str:
        cell_context = getattr(self, "current_cell_id", "unknown")
        short_cell = str(cell_context)[:8] if cell_context != "unknown" else "unk"

        return f"{var_name}_{short_cell}_{lineno}"

    def _handle_iterable(self, iterable_node: ast.AST) -> str:
        """Visits the iterable and returns the ID of the resulting state node."""
        base_name = get_base_name(iterable_node)
        if isinstance(iterable_node, ast.Name) and base_name:
            return self._state_stack.get_last_variable_version(base_name)

        cell_id = getattr(self, "current_cell_id", "unk")
        iter_state_node = f"iter_val_{str(cell_id)[:8]}_{iterable_node.lineno}_{iterable_node.col_offset}"
        self._add_node(iter_state_node, NODE_TYPES["INTERMEDIATE"])

        original_var = self._current_state.current_variable
        self._current_state.set_current_variable(iter_state_node)
        self.visit(iterable_node)
        self._current_state.set_current_variable(original_var)

        return iter_state_node

    def _add_node(self, node_name: str, node_type: int, code: str = "", label: str | None = None) -> None:
        cell_context = getattr(self, "current_cell_id", "unknown_cell")

        if not label:
            clean_name = re.sub(r"_[a-z0-9]{8}_\d+(_\d+)?.*$", "", node_name)
            label = clean_name.replace("_", " ")

        if self._current_state.current_variable and code and node_type in (NODE_TYPES.get("CALL", 9), NODE_TYPES["LOOP"]):
            var_node = self._current_state.get_node(self._current_state.current_variable) if self._current_state._G.has_node(self._current_state.current_variable) else None
            if var_node:
                history = var_node.get("transform_history", [])
                history.append(code)
                self._current_state._G.nodes[self._current_state.current_variable]["transform_history"] = history

        self._current_state.add_node(node_name, node_type, code=code, cell_id=cell_context, label=label)

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
        end_col_offset: int = -1,
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
            cell_id=cell_context,
        )

    def _generate_context_name(self, prefix: str, node: ast.AST) -> str:
        self._scope_counter += 1
        cell_id = getattr(self, "current_cell_id", "unknown_cell")
        return f"{prefix}_{cell_id}_{node.lineno}_{node.col_offset}_{self._scope_counter}"

    def _process_generic_call(self, node: ast.Call, caller_object_name: str | None, function_name: str | None) -> None:
        """Fallback processor for unknown function calls to enforce bipartite routing."""
        raw_code = ast.get_source_segment(self.current_cell_source, node) or "call"

        label = function_name.replace("_", " ") if function_name else "call"

        cell_id = getattr(self, "current_cell_id", "unk")
        op_node = f"call_{function_name}_{str(cell_id)[:8]}_{node.lineno}_{node.col_offset}"

        self._add_node(op_node, NODE_TYPES.get("CALL", 9), code=raw_code, label=label)

        if caller_object_name and caller_object_name != function_name:
            prev_version = self._state_stack.get_last_variable_version(caller_object_name)
            prev_version = prev_version if prev_version else self._current_state.last_variable

            if prev_version:
                self._add_edge(prev_version, op_node, label, EDGE_TYPES["CALLER"], raw_code=raw_code, lineno=node.lineno)

        original_var = self._current_state.current_variable
        self._current_state.set_current_variable(op_node)

        self._visit_call_args(node, op_node)

        self._current_state.set_current_variable(original_var)

        if original_var:
            self._add_edge(op_node, original_var, "output", EDGE_TYPES["OMITTED"], raw_code=raw_code, lineno=node.lineno)

        self._current_state.set_last_variable(op_node)

    def _visit_call_args(self, node: ast.Call, op_node: str) -> None:
        """Visits call arguments while natively detecting and bridging Higher-Order Function (HOF) pointers."""
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in self._state_stack.functions:
                func_def_node = self._state_stack.functions[arg.id]["node"]
                self._add_edge(
                    source=func_def_node,
                    target=op_node,
                    label="function pointer",
                    edge_type=EDGE_TYPES.get("INPUT", 0),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=getattr(node, "end_lineno", -1),
                    end_col_offset=getattr(node, "end_col_offset", -1),
                )
            self.visit(arg)

        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Name) and keyword.value.id in self._state_stack.functions:
                func_def_node = self._state_stack.functions[keyword.value.id]["node"]
                self._add_edge(
                    source=func_def_node,
                    target=op_node,
                    label="function pointer",
                    edge_type=EDGE_TYPES.get("INPUT", 0),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=getattr(node, "end_lineno", -1),
                    end_col_offset=getattr(node, "end_col_offset", -1),
                )
            self.visit(keyword.value)
