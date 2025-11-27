from typing import Optional

from ApexDAG.util.logging import setup_logging
from ApexDAG.sca.constants import VERBOSE
from ApexDAG.state.state import State


class Stack:
    def __init__(self) -> None:
        self._logger = setup_logging("state.Stack", VERBOSE)

        self.imported_names = {}
        self.import_from_modules = {}
        self.classes = {}
        self.functions = {}
        self.instances = {} # Maps instance names to class names
        self.branches = []

        self._current_state = "module"
        self._state = {"module": State("module")}

    def __contains__(self, context: str) -> bool:
        return context in self._state

    def create_child_state(
        self, context: str = None, parent_context: Optional[str] = None
    ) -> None:
        if context in self._state:
            raise ValueError(f"State {context} already exists")

        if parent_context is not None and parent_context not in self._state:
            raise ValueError(f"Parent state {parent_context} does not exist")

        self._state[context] = State(context, parent_context)
        self._current_state = context

    def add_class_instance(self, instance: str, class_name: str):
        self.instances[instance] = class_name
        
    def restore_state(self, context: str) -> None:
        if context in self._state:
            self._current_state = context
        else:
            raise ValueError(f"No state {context} to restore")

    def restore_parent_state(self) -> None:
        parent_context = self._state[self._current_state].parent_context
        if parent_context not in self._state:
            raise ValueError(f"No parent state {parent_context} to restore")

        self.restore_state(parent_context)

    def merge_states(self, base_context: str, args: list[tuple]) -> None:
        if base_context not in self._state:
            raise ValueError(f"No state {base_context} to merge")

        base_state = self._state[base_context]
        base_state.merge(*args)
        

        self._current_state = base_context
        
    def merge_class_method_state(
        self, 
        base_context: str, 
        method_state: "State",
        method_name: str,
        edge_type: str,
        instance_name: str = None,
        base_name: str = None,
    ) -> None:
        """
        Merge a class method's state back into the caller context.
        Handles instance variable updates and filters out method-local variables.
        
        Args:
            base_context: The caller's context to merge back into
            method_state: The method's state to merge from
            method_name: Name of the method being merged
            edge_type: Edge type for the merge operation
            instance_name: Name of the instance variable (if applicable)
        """
        if base_context not in self._state:
            raise ValueError(f"No state {base_context} to merge")

        base_state = self._state[base_context]
        
        # Get method's graph
        method_graph = method_state.get_graph()
        
        # Build node mapping for self.* to instance_name.*
        node_mapping = {}
        if instance_name:
            for node, node_data in method_graph.nodes(data=True):
                if node.startswith("self"):
                    # Map self.attribute to instance_name.attribute in caller context
                    attr_name = node.replace("self", "")
                    instance_attr = f"{instance_name}{attr_name}"
                    node_mapping[node] = instance_attr
                    
                    if self._state['module'].current_target not in self._state['module'].variable_versions:
                        self._state['module'].variable_versions[self._state['module'].current_target] = []
                    self._state['module'].variable_versions[self._state['module'].current_target].append(instance_attr)
        
        # Create a new graph with renamed nodes for merging
        import networkx as nx
        renamed_graph = nx.MultiDiGraph()
        
        # Add all nodes with mapping applied
        for node, node_attrs in method_graph.nodes(data=True):
            mapped_node = node_mapping.get(node, node)
            # Update label if node was mapped
            attrs_copy = node_attrs.copy()
            if node in node_mapping:
                attrs_copy["label"] = mapped_node
            renamed_graph.add_node(mapped_node, **attrs_copy)
        
        # Add all edges with mapping applied to both source and target
        for source, target, key, edge_data in method_graph.edges(keys=True, data=True):
            mapped_source = node_mapping.get(source, source)
            mapped_target = node_mapping.get(target, target)
            renamed_graph.add_edge(mapped_source, mapped_target, key=key, **edge_data)
        
        # Update variable versions - create mapped version
        new_variable_versions = {}
        for var_name, versions in method_state.variable_versions.items():
            if var_name == "self":
                continue  # Skip self parameter
            
            mapped_versions = []
            for version in versions:
                if version in node_mapping:
                    mapped_versions.append(node_mapping[version])
                else:
                    mapped_versions.append(version)
            
            # Determine the variable name in base context
            if any(v in node_mapping for v in versions):
                # This is an instance attribute
                attr_name = var_name.replace("self", "").split(":")[0]
                base_var_name = f"{instance_name}{attr_name}" if instance_name else var_name
                method_state.variable_versions[base_name].append(
                    base_var_name
                )
            else:
                base_var_name = var_name
            
            new_variable_versions[base_var_name] = mapped_versions
        
        # Now merge variable versions following the original merge logic
        for var_name, versions in new_variable_versions.items():
            if var_name in base_state.variable_versions:
                # Variable exists in base, connect last version to first new version
                last_var = base_state.variable_versions[var_name][-1]
                new_var = versions[0]
                base_state.add_edge(last_var, new_var, method_name, edge_type)
                
            else:
                # New variable, just add it
                base_state.variable_versions[var_name] = versions
        
        # Compose the renamed graph into base state
        import networkx as nx
        print("Base graph nodes before merge:", base_state._G.nodes(data=True))
        print("Renamed graph nodes to merge:", renamed_graph.nodes(data=True))
        base_state._G = nx.compose(base_state._G, renamed_graph)
        
        self._current_state = base_context

    def get_current_state(self) -> State:
        return self._state[self._current_state]
