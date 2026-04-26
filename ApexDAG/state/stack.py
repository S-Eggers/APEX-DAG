import logging
from contextlib import contextmanager

import networkx as nx

from ApexDAG.state.state import State
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class Stack:
    def __init__(self) -> None:

        self.imported_names = {}
        self.import_from_modules = {}
        self.classes = {}
        self.functions = {}
        self.instances = {}  # Maps instance names to class names
        self.branches = []
        self._nested = False

        self._current_state = "module"
        self._state = {"module": State("module")}

    def __contains__(self, context: str) -> bool:
        return context in self._state

    @property
    def nested(self) -> bool:
        return self._nested

    @nested.setter
    def nested(self, nested: bool) -> None:
        self._nested = nested

    def create_child_state(
        self, context: str | None = None, parent_context: str | None = None
    ) -> None:
        if context in self._state:
            raise ValueError(f"State {context} already exists")

        if parent_context is not None and parent_context not in self._state:
            raise ValueError(f"Parent state {parent_context} does not exist")

        self._state[context] = State(context, parent_context)
        # print(f"Created new state: {context} with parent {parent_context}")
        self._current_state = context

    @contextmanager
    def scope(self, context: str, parent_context: str | None = None):
        """
        Guarantees safe state transitions. Automatically builds the child state,
        yields it to the AST visitor, and securely restores the parent context on exit,
        even if an exception occurs during traversal.
        """
        self.create_child_state(context, parent_context)
        try:
            yield self.get_current_state()
        finally:
            self.restore_state(parent_context)

    def add_class_instance(self, instance: str, class_name: str) -> None:
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

    def merge_states(
        self, base_context: str, args: list[tuple], cell_id: str = "unknown_cell"
    ) -> None:
        if base_context not in self._state:
            raise ValueError(f"No state {base_context} to merge")

        base_state = self._state[base_context]
        base_state.merge(args, cell_id=cell_id)

        self._current_state = base_context

    def merge_class_method_state(
        self,
        base_context: str,
        method_state: "State",
        method_name: str,
        edge_type: str,
        instance_name: str | None = None,
        base_name: str | None = None,
        cell_id: str = "unknown_cell",
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
            for node, _node_data in method_graph.nodes(data=True):
                if node.startswith("self"):
                    # Map self.attribute to instance_name.attribute in caller context
                    attr_name = node.replace("self", "")
                    instance_attr = f"{instance_name}{attr_name}"
                    node_mapping[node] = instance_attr

                    if (
                        self._state["module"].current_target
                        not in self._state["module"].variable_versions
                    ):
                        self._state["module"].variable_versions[
                            self._state["module"].current_target
                        ] = []
                    self._state["module"].variable_versions[
                        self._state["module"].current_target
                    ].append(instance_attr)

        # Create a new graph with renamed nodes for merging
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
                base_var_name = (
                    f"{instance_name}{attr_name}" if instance_name else var_name
                )
                method_state.variable_versions[base_name].append(base_var_name)
            else:
                base_var_name = var_name

            new_variable_versions[base_var_name] = mapped_versions

        for var_name, versions in new_variable_versions.items():
            if var_name in base_state.variable_versions:
                last_var = base_state.variable_versions[var_name][-1]
                new_var = versions[0]

                base_state.add_edge(
                    source=last_var,
                    target=new_var,
                    label=method_name,
                    edge_type=edge_type,
                    raw_code=method_name,
                    cell_id=cell_id,
                )
            else:
                base_state.variable_versions[var_name] = versions

        # Compose the renamed graph into base state
        base_state._G = nx.compose(base_state._G, renamed_graph)

        self._current_state = base_context

    def get_current_state(self) -> State:
        return self._state[self._current_state]

    def get_last_variable_version(
        self, variable: str, max_depth: int = 99
    ) -> str | None:
        current_state = self.get_current_state()
        if variable in current_state.variable_versions:
            return current_state.variable_versions.get(variable)[-1]

        if current_state.parent_context and max_depth > 0:
            active_context = current_state.context
            self.restore_parent_state()
            variable_version = self.get_last_variable_version(
                variable, max_depth=max_depth - 1
            )
            self.restore_state(active_context)
            return variable_version
        return None

    def import_accessible(self, name: str, max_depth: int = 99) -> bool:
        if name in self.import_from_modules or name in self.imported_names:
            return True
        current_state = self.get_current_state()
        if current_state.parent_context and max_depth > 0:
            active_context = current_state.context
            self.restore_parent_state()
            reachable = self.import_accessible(name, max_depth=max_depth - 1)
            self.restore_state(active_context)
            return reachable
        return False

    def class_accessible(self, name: str, max_depth: int = 99) -> bool:
        if name in self.classes:
            return True
        current_state = self.get_current_state()
        if current_state.parent_context and max_depth > 0:
            active_context = current_state.context
            self.restore_parent_state()
            reachable = self.class_accessible(name, max_depth=max_depth - 1)
            self.restore_state(active_context)
            return reachable
        return False

    def function_accessible(self, name: str, max_depth: int = 99) -> bool:
        if name in self.functions:
            return True
        current_state = self.get_current_state()
        if current_state.parent_context and max_depth > 0:
            active_context = current_state.context
            self.restore_parent_state()
            reachable = self.function_accessible(name, max_depth=max_depth - 1)
            self.restore_state(active_context)
            return reachable
        return False
