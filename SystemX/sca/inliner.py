import ast

from SystemX.sca.ast_utils import (
    get_base_name,
    tokenize_method,
)
from SystemX.sca.constants import EDGE_TYPES, NODE_TYPES

from .types.graph_context import GraphContext
from .types.inlining_policy import InliningPolicy


class CallInliner:
    """Orchestrates function and class method inlining for the DataFlow graph."""

    def __init__(self, context: GraphContext, policy: InliningPolicy) -> None:
        self.ctx = context
        self.stack = context.state_stack
        self.policy = policy

    def process_class(self, node: ast.Call, caller_object_name: str, tokens: str, is_instance: bool) -> None:
        parts = tokens.split(".") if tokens else []
        if is_instance:
            class_name = self.stack.instances.get(caller_object_name, caller_object_name)
            method_name = parts[-1]
        else:
            class_name = parts[0] if parts else caller_object_name
            method_name = parts[-1] if len(parts) > 1 else "__init__"

        self.stack.add_class_instance(self.ctx.current_state.current_target, class_name)
        full_method_name = f"{class_name}.{method_name}"

        if self.policy.should_inline_class_method(class_name, method_name, self.stack):
            self._execute_inline(
                node=node,
                target_func=full_method_name,
                tokens=tokens,
                caller_obj=caller_object_name,
                is_instance=is_instance,
                class_name=class_name,
            )
        else:
            self._execute_class_fallback(node, caller_object_name, tokens)

    def process_function(self, node: ast.Call, function_name: str) -> None:
        if self.policy.should_inline_function(function_name, self.stack):
            self._execute_inline(node, function_name, function_name)
        else:
            self._execute_function_fallback(node, function_name)

    def _execute_inline(
        self,
        node: ast.Call,
        target_func: str,
        tokens: str,
        caller_obj: str | None = None,
        is_instance: bool = False,
        class_name: str | None = None,
    ) -> None:
        is_class_method = bool(class_name)
        caller_return_var = self.ctx.current_state.current_variable
        current_context = self.ctx.current_state.context
        method_context = self.stack.functions[target_func]["context"]

        param_to_caller = self._map_arguments(node, target_func, is_class_method)

        self.stack.restore_state(method_context)
        self.ctx.current_state = self.stack.get_current_state()

        instance_version = None
        if class_name and caller_obj:
            instance_version = self._bind_instance(node, caller_obj, is_instance, caller_return_var, tokens)
            self._bind_self(node, instance_version)

        self._bind_parameters(node, param_to_caller, is_class_method)
        self._bind_returns(node, target_func, caller_return_var, is_class_method)

        cell_context = getattr(self.ctx, "current_cell_id", "unknown_cell")
        tokenized_tokens = tokenize_method(tokens, self.stack.imported_names, self.stack.import_from_modules)

        if class_name and instance_version:
            self.stack.merge_class_method_state(
                current_context,
                self.ctx.current_state,
                tokenized_tokens,
                EDGE_TYPES["CLASS_CALL"],
                instance_name=instance_version,
                base_name=get_base_name(instance_version),
                cell_id=cell_context,
            )
        else:
            self.stack.merge_states(
                current_context,
                [(self.ctx.current_state, tokenized_tokens, EDGE_TYPES["FUNCTION_CALL"])],
                cell_id=cell_context,
            )

        self.ctx.current_state = self.stack.get_current_state()

    def _map_arguments(self, node: ast.Call, target_func: str, is_class_method: bool) -> dict[str, str]:
        f_args = self.stack.functions[target_func]["args"]["args"]
        if is_class_method and f_args and f_args[0] == "self":
            f_args = f_args[1:]

        param_to_caller: dict[str, str] = {}
        for i, arg_node in enumerate(node.args):
            if i < len(f_args):
                param_name = f_args[i]
                caller_base = get_base_name(arg_node)
                if caller_base:
                    caller_version = self.stack.get_last_variable_version(caller_base)
                    if caller_version:
                        param_to_caller[param_name] = caller_version

        for kw in node.keywords:
            if kw.arg:
                caller_base = get_base_name(kw.value)
                if caller_base:
                    caller_version = self.stack.get_last_variable_version(caller_base)
                    if caller_version:
                        param_to_caller[kw.arg] = caller_version

        return param_to_caller

    def _bind_instance(
        self,
        node: ast.Call,
        caller_obj: str,
        is_instance: bool,
        caller_return_var: str | None,
        tokens: str,
    ) -> str | None:
        if is_instance:
            return self.stack.get_last_variable_version(caller_obj)

        if caller_obj not in self.stack.classes:
            return None

        class_node = self.stack.classes[caller_obj][0]
        self.ctx.add_node(class_node, NODE_TYPES["CLASS"])

        raw_code = ast.get_source_segment(self.ctx.current_cell_source, node.func) or tokens
        label = tokenize_method(raw_code, self.stack.imported_names, self.stack.import_from_modules)

        if caller_return_var:
            self.ctx.add_edge(
                source=class_node,
                target=caller_return_var,
                label=label,
                edge_type=EDGE_TYPES["CALLER"],
                raw_code=raw_code,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=getattr(node, "end_lineno", -1),
                end_col_offset=getattr(node, "end_col_offset", -1),
            )
        return caller_return_var

    def _bind_self(self, node: ast.Call, instance_version: str | None) -> None:
        if instance_version and "self" in self.ctx.current_state.variable_versions:
            self_param_node = self.ctx.current_state.variable_versions["self"][0]
            last_self_node = self.stack.get_last_variable_version("self")

            for src, tgt in [
                (instance_version, self_param_node),
                (last_self_node, instance_version),
            ]:
                if src and tgt:
                    self.ctx.add_edge(
                        source=src,
                        target=tgt,
                        label="self_binding",
                        edge_type=EDGE_TYPES["OMITTED"],
                        raw_code="self_binding",
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        end_lineno=getattr(node, "end_lineno", -1),
                        end_col_offset=getattr(node, "end_col_offset", -1),
                    )

    def _bind_parameters(self, node: ast.Call, param_to_caller: dict[str, str], is_class_method: bool) -> None:
        edge_type = EDGE_TYPES["INPUT"] if is_class_method else EDGE_TYPES["FUNCTION_CALL"]
        for param_name, caller_node in param_to_caller.items():
            if param_name in self.ctx.current_state.variable_versions:
                func_param_node = self.ctx.current_state.variable_versions[param_name][0]
                self.ctx.add_edge(
                    source=caller_node,
                    target=func_param_node,
                    label="arg_pass",
                    edge_type=edge_type,
                    raw_code="arg_pass",
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=getattr(node, "end_lineno", -1),
                    end_col_offset=getattr(node, "end_col_offset", -1),
                )

    def _bind_returns(self, node: ast.Call, target_func: str, caller_return_var: str | None, is_class_method: bool) -> None:
        if not caller_return_var:
            return

        edge_type = EDGE_TYPES["OMITTED"] if is_class_method else EDGE_TYPES["FUNCTION_CALL"]
        return_nodes = self.stack.functions[target_func].get("return_nodes", [])
        for ret_node in return_nodes:
            self.ctx.add_edge(
                source=ret_node,
                target=caller_return_var,
                label="return",
                edge_type=edge_type,
                raw_code="return",
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=getattr(node, "end_lineno", -1),
                end_col_offset=getattr(node, "end_col_offset", -1),
            )

    def _execute_class_fallback(self, node: ast.Call, caller_obj: str, tokens: str) -> None:
        class_node = self.stack.classes[caller_obj][0]
        self.ctx.add_node(class_node, NODE_TYPES["CLASS"])

        raw_code = ast.get_source_segment(self.ctx.current_cell_source, node.func) or tokens
        label = tokenize_method(raw_code, self.stack.imported_names, self.stack.import_from_modules)

        cell_context = getattr(self.ctx, "current_cell_id", "unk")
        op_node_name = f"init_{label}_{str(cell_context)[:8]}_{node.lineno}_{node.col_offset}"

        self.ctx.add_node(op_node_name, NODE_TYPES.get("CALL", 9), code=raw_code)

        self.ctx.add_edge(
            source=class_node,
            target=op_node_name,
            label=label,
            edge_type=EDGE_TYPES["CALLER"],
            raw_code=raw_code,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=getattr(node, "end_lineno", -1),
            end_col_offset=getattr(node, "end_col_offset", -1),
        )

        original_var = self.ctx.current_state.current_variable
        self.ctx.current_state.set_current_variable(op_node_name)

        for arg in node.args:
            self.ctx.visit(arg)
        for keyword in node.keywords:
            self.ctx.visit(keyword.value)

        self.ctx.current_state.set_current_variable(original_var)

        if self.ctx.current_state.current_variable:
            self.ctx.add_edge(
                source=op_node_name,
                target=self.ctx.current_state.current_variable,
                label="output",
                edge_type=EDGE_TYPES["OMITTED"],
                raw_code=raw_code,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=getattr(node, "end_lineno", -1),
                end_col_offset=getattr(node, "end_col_offset", -1),
            )

        self.ctx.current_state.set_last_variable(class_node)

    def _execute_function_fallback(self, node: ast.Call, function_name: str) -> None:
        func_def_node = self.stack.functions[function_name]["node"]

        cell_context = getattr(self.ctx, "current_cell_id", "unk")
        op_node_name = f"call_{function_name}_{str(cell_context)[:8]}_{node.lineno}_{node.col_offset}"

        self.ctx.add_node(op_node_name, NODE_TYPES.get("CALL", 9), code=function_name)

        label = tokenize_method(function_name, self.stack.imported_names, self.stack.import_from_modules)

        self.ctx.add_edge(
            source=func_def_node,
            target=op_node_name,
            label=label,
            edge_type=EDGE_TYPES["FUNCTION_CALL"],
            raw_code=function_name,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=getattr(node, "end_lineno", -1),
            end_col_offset=getattr(node, "end_col_offset", -1),
        )

        original_var = self.ctx.current_state.current_variable
        self.ctx.current_state.set_current_variable(op_node_name)

        for arg in node.args:
            self.ctx.visit(arg)
        for keyword in node.keywords:
            self.ctx.visit(keyword.value)

        self.ctx.current_state.set_current_variable(original_var)

        if self.ctx.current_state.current_variable:
            self.ctx.add_edge(
                source=op_node_name,
                target=self.ctx.current_state.current_variable,
                label="output",
                edge_type=EDGE_TYPES["OMITTED"],
                raw_code=function_name,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=getattr(node, "end_lineno", -1),
                end_col_offset=getattr(node, "end_col_offset", -1),
            )
