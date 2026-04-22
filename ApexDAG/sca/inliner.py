import ast
from ApexDAG.sca.constants import EDGE_TYPES, NODE_TYPES
from ApexDAG.sca import flatten_list 


class CallInliner:
    """
    Orchestrates function and class method inlining for the DataFlow graph.
    """
    def __init__(self, visitor: "PythonDataFlowGraph"):
        self.v = visitor
        self.stack = visitor._state_stack

    def process_class(self, node: ast.Call, caller_object_name: str, tokens: str, is_instance: bool):
        parts = tokens.split(".") if tokens else []
        if is_instance:
            class_name = self.stack.instances.get(caller_object_name, caller_object_name)
            method_name = parts[-1]
        else:
            class_name = parts[0] if parts else caller_object_name
            method_name = parts[-1] if len(parts) > 1 else "__init__"

        self.stack.add_class_instance(self.v._current_state.current_target, class_name)
        full_method_name = f"{class_name}.{method_name}"
        
        should_inline = (
            self.v._replace_dataflow 
            and class_name in self.stack.classes
            and full_method_name in self.stack.functions
            and not self.stack.functions[full_method_name]["is_recursive"]
        )

        if should_inline:
            self._execute_inline(node, full_method_name, tokens, caller_object_name, is_instance, class_name)
        else:
            self._execute_class_fallback(node, caller_object_name, tokens)

    def process_function(self, node: ast.Call, function_name: str):
        should_inline = (
            self.v._replace_dataflow
            and function_name in self.stack.functions
            and not self.stack.functions[function_name]["is_recursive"]
        )

        if should_inline:
            self._execute_inline(node, function_name, function_name)
        else:
            self._execute_function_fallback(node, function_name)

    def _execute_inline(self, node: ast.Call, target_func: str, tokens: str, 
                        caller_obj: str = None, is_instance: bool = False, class_name: str = None):
        
        caller_return_var = self.v._current_state.current_variable
        current_context = self.v._current_state.context
        method_context = self.stack.functions[target_func]["context"]
        
        # 1. Map Arguments
        param_to_caller = self._map_arguments(node, target_func, is_class_method=bool(class_name))
        
        # 2. Switch Context
        self.stack.restore_state(method_context)
        self.v._current_state = self.stack.get_current_state()
        
        # 3. Bind everything inside the method scope
        instance_version = None
        if class_name:
            instance_version = self._bind_instance(node, caller_obj, is_instance, caller_return_var, tokens)
            self._bind_self(node, instance_version)
            
        self._bind_parameters(node, param_to_caller)
        self._bind_returns(node, target_func, caller_return_var)
        
        # 4. Merge back to parent
        cell_context = getattr(self.v, "current_cell_id", "unknown_cell")
        if class_name:
            self.stack.merge_class_method_state(
                current_context,
                self.v._current_state,
                self.v._tokenize_method(tokens),
                EDGE_TYPES["CLASS_CALL"],
                instance_name=instance_version,
                base_name=self.v._get_base_name(instance_version),
                cell_id=cell_context
            )
        else:
            self.stack.merge_states(
                current_context,
                [(self.v._current_state, self.v._tokenize_method(tokens), EDGE_TYPES["FUNCTION_CALL"])],
                cell_id=cell_context
            )
            
        self.v._current_state = self.stack.get_current_state()

    def _map_arguments(self, node: ast.Call, target_func: str, is_class_method: bool) -> dict:
        f_args = self.stack.functions[target_func]["args"]["args"]
        if is_class_method and f_args and f_args[0] == "self":
            f_args = f_args[1:] # Skip 'self' for mapping caller args
            
        param_to_caller = {}
        for i, arg_node in enumerate(node.args):
            if i < len(f_args):
                param_name = f_args[i]
                caller_base = self.v._get_base_name(arg_node)
                if caller_base:
                    caller_version = self.v._get_last_variable_version(caller_base)
                    if caller_version:
                        param_to_caller[param_name] = caller_version

        for kw in node.keywords:
            param_name = kw.arg
            caller_base = self.v._get_base_name(kw.value)
            if caller_base:
                caller_version = self.v._get_last_variable_version(caller_base)
                if caller_version:
                    param_to_caller[param_name] = caller_version
                    
        return param_to_caller

    def _bind_instance(self, node: ast.Call, caller_obj: str, is_instance: bool, caller_return_var: str, tokens: str):
        if is_instance:
            return self.v._get_last_variable_version(caller_obj)
        
        # Constructor
        class_node = self.stack.classes[caller_obj][0]
        self.v._add_node(class_node, NODE_TYPES["CLASS"])
        
        raw_code = ast.get_source_segment(self.v.current_cell_source, node.func) or tokens
        label = self.v._tokenize_method(raw_code)
        
        if caller_return_var:
            self.v._add_edge(
                source=class_node, target=caller_return_var,
                label=label, edge_type=EDGE_TYPES["CALLER"], raw_code=raw_code,
                lineno=node.lineno, col_offset=node.col_offset,
                end_lineno=node.end_lineno, end_col_offset=node.end_col_offset
            )
        return caller_return_var

    def _bind_self(self, node: ast.Call, instance_version: str):
        if instance_version and 'self' in self.v._current_state.variable_versions:
            self_param_node = self.v._current_state.variable_versions['self'][0]
            last_self_node = self.v._get_last_variable_version("self")
            
            for src, tgt in [(instance_version, self_param_node), (last_self_node, instance_version)]:
                self.v._add_edge(
                    source=src, target=tgt,
                    label="self_binding", edge_type=EDGE_TYPES["OMITTED"], raw_code="self_binding",
                    lineno=node.lineno, col_offset=node.col_offset,
                    end_lineno=node.end_lineno, end_col_offset=node.end_col_offset
                )

    def _bind_parameters(self, node: ast.Call, param_to_caller: dict):
        for param_name, caller_node in param_to_caller.items():
            if param_name in self.v._current_state.variable_versions:
                func_param_node = self.v._current_state.variable_versions[param_name][0]
                self.v._add_edge(
                    source=caller_node, target=func_param_node,
                    label="arg_pass", edge_type=EDGE_TYPES["INPUT"] if "CLASS" in str(type(self)) else EDGE_TYPES["FUNCTION_CALL"],
                    raw_code="arg_pass",
                    lineno=node.lineno, col_offset=node.col_offset,
                    end_lineno=node.end_lineno, end_col_offset=node.end_col_offset
                )

    def _bind_returns(self, node: ast.Call, target_func: str, caller_return_var: str):
        if not caller_return_var: return
        return_nodes = self.stack.functions[target_func].get("return_nodes", [])
        for ret_node in return_nodes:
            self.v._add_edge(
                source=ret_node, target=caller_return_var,
                label="return", edge_type=EDGE_TYPES["OMITTED"] if "CLASS" in str(type(self)) else EDGE_TYPES["FUNCTION_CALL"],
                raw_code="return",
                lineno=node.lineno, col_offset=node.col_offset,
                end_lineno=node.end_lineno, end_col_offset=node.end_col_offset
            )

    def _execute_class_fallback(self, node: ast.Call, caller_obj: str, tokens: str):
        class_node = self.stack.classes[caller_obj][0]
        self.v._add_node(class_node, NODE_TYPES["CLASS"])
        
        raw_code = ast.get_source_segment(self.v.current_cell_source, node.func) or tokens
        label = self.v._tokenize_method(raw_code)
        
        if self.v._current_state.current_variable:
            self.v._add_edge(
                source=class_node, target=self.v._current_state.current_variable,
                label=label, edge_type=EDGE_TYPES["CALLER"], raw_code=raw_code,
                lineno=node.lineno, col_offset=node.col_offset,
                end_lineno=node.end_lineno, end_col_offset=node.end_col_offset
            )
        
        for arg in node.args:
            if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                self.v._process_name_attr_sub_args(node, arg)
        
        self.v._current_state.set_last_variable(class_node)

    def _execute_function_fallback(self, node: ast.Call, function_name: str):
        func_def_node = self.stack.functions[function_name]["node"]
        if self.v._current_state.current_variable:
            self.v._add_edge(
                source=func_def_node, target=self.v._current_state.current_variable,
                label=self.v._tokenize_method(function_name), edge_type=EDGE_TYPES["FUNCTION_CALL"],
                raw_code=function_name,
                lineno=node.lineno, col_offset=node.col_offset,
                end_lineno=node.end_lineno, end_col_offset=node.end_col_offset
            )
            for arg in node.args:
                if isinstance(arg, (ast.Name, ast.Attribute, ast.Subscript)):
                    arg_names = self.v._get_names(arg)
                    flat_args = flatten_list(arg_names) if arg_names else None
                    arg_name = flat_args[0] if isinstance(flat_args, list) else flat_args
                    
                    if arg_name:
                        arg_version = self.v._get_last_variable_version(arg_name)
                        if arg_version:
                            self.v._add_edge(
                                source=arg_version, target=self.v._current_state.current_variable,
                                label=self.v._tokenize_method(str(arg_name)), edge_type=EDGE_TYPES["INPUT"],
                                raw_code=str(arg_name),
                                lineno=node.lineno, col_offset=node.col_offset,
                                end_lineno=node.end_lineno, end_col_offset=node.end_col_offset
                            )
