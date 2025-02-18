import networkx as nx
from typing import Optional
from networkx import Graph, MultiDiGraph

from ApexDAG.util.logging import setup_logging
from ApexDAG.sca.constants import EDGE_TYPES, NODE_TYPES, VERBOSE


class State:
    def __init__(self, name: str, parent_context: Optional[str] = None) -> None:
        self._logger = setup_logging("state.State", VERBOSE)
        self.edge_for_current_target: dict = {}
        self.variable_versions: dict = {}
        self.imported_names: dict = {}
        self.import_from_modules: dict = {}
        self.classes:dict = {}
        self.functions:dict = {}
        self.current_target: str = None
        self.current_variable: str = None
        self.last_variable: str = None
        self.payload: dict = None
        self.context: str = name
        self.parent_context: Optional[str] = parent_context
        self._G: MultiDiGraph = MultiDiGraph()

    def __getitem__(self, name):
        match name:
            case "current_variable":
                return self.current_variable
            case "current_target":
                return self.current_target
            case "last_variable":
                return self.last_variable
            case "payload":
                return self.payload
            case "context":
                return self.context
            case "parent_context":
                return self.parent_context
            case "edge_for_current_target":
                return self.edge_for_current_target
            case "variable_versions":
                return self.variable_versions
            case "imported_names":
                return self.imported_names
            case "import_from_modules":
                return self.import_from_modules
            case "classes":
                return self.classes
            case "functions":
                return self.functions
            case "_G":
                return self._G
            case _:
                raise ValueError(f"Attribute {name} not a valid attribute")

    def __setitem__(self, name, value):
        match name:
            case "current_variable":
                self.current_variable = value
            case "current_target":
                self.current_target = value
            case "last_variable":
                self.last_variable = value
            case "payload":
                self.payload = value
            case "context":
                self.context = value
            case "parent_context":
                self.parent_context = value
            case "edge_for_current_target":
                self.edge_for_current_target = value
            case "variable_versions":
                self.variable_versions = value
            case "imported_names":
                self.imported_names = value
            case "import_from_modules":
                self.import_from_modules = value
            case "classes":
                self.classes = value
            case "functions":
                self.functions = value
            case "_G":
                self._G = value
            case _:
                raise ValueError(f"Attribute {name} not a valid attribute")

    def set_current_variable(self, value: str):
        self.current_variable = value

    def set_current_target(self, value: str):
        self.current_target = value

    def set_last_variable(self, value: str):
        self.last_variable = value

    def merge(self, *args: tuple):
        new_variable_versions = {key: value[:] for key, value in self.variable_versions.items()}
        branched_variables = {}
        looped_variables = {}

        for state, var_edge, edge_type in args:
            is_loop = edge_type == EDGE_TYPES["LOOP"]
            for key, value in state["variable_versions"].items():
                if key in self.variable_versions:
                    last_var = self.variable_versions[key][-1]
                    new_var = value[0]
                    self.add_edge(last_var, new_var, var_edge, edge_type)
                    new_variable_versions[key] += value
                else:
                    new_variable_versions[key] = value

                if key in branched_variables:
                    branched_variables[key].append(value[-1])
                else:
                    branched_variables[key] = [value[-1]]

                if is_loop and key in self.variable_versions:
                    last_var = self.variable_versions[key][-1]
                    looped_variables[key] = (last_var, value[-1])

            self._G = nx.compose(self._G, state["_G"])

        for variable, loop in looped_variables.items():
            var_name = f"loop_{'_'.join(loop)}"
            self.add_node(var_name, NODE_TYPES["LOOP"])
            new_variable_versions[variable].append(var_name)
            self.add_edge(loop[1], var_name, "end_loop", EDGE_TYPES["LOOP"])
            self.add_edge(var_name, loop[0], "restart_loop", EDGE_TYPES["LOOP"])

        for variable, branches in branched_variables.items():
            if len(branches) > 1:
                var_name = f"branch_{'_'.join(branches)}"
                self.add_node(var_name, NODE_TYPES["IF"])
                new_variable_versions[variable].append(var_name)
                for node in branches:
                    self.add_edge(node, var_name, "end_if", EDGE_TYPES["BRANCH"])

        self.variable_versions = new_variable_versions

    def copy_graph(self) -> Graph:
        return self._G.copy()

    def set_graph(self, G) -> None:
        self._G = G

    def get_graph(self) -> MultiDiGraph:
        return self._G

    def get_node(self, node_identifier: str):
        return self._G.nodes[node_identifier]

    def remove_node(self, node_identifier: str) -> None:
        self._G.remove_node(node_identifier)

    def node_iterator(self):
        for node in self._G.nodes:
            yield node
     
    def adjacent_node_iterator(self, node_identifier: str):
        for node in self._G[node_identifier]:
            yield node

    def add_node(
        self,
        node: str,
        node_type: int
     ):
        if node not in self._G:
            self._G.add_node(node, label=node, node_type=node_type)
        else:
            self._logger.debug("Node %s already exists in the graph", node)

    def node_degree(self, node_identifier: str):
        return {
            "in": self._G.out_degree(node_identifier),
            "out": self._G.in_degree(node_identifier),
        }

    def predecessor_node_iterator(self, node_identifier: str):
        for node in self._G.predecessors(node_identifier):
            yield node

    def successor_node_iterator(self, node_identifier: str):
        for node in self._G.successors(node_identifier):
            yield node

    def has_edge(self, source: str, target: str, key: Optional[str]=None) -> bool:
        if key:
            return self._G.has_edge(source, target, key=key)
        else:
            return self._G.has_edge(source, target)

    def get_edge_iterator(self, source: str, target: str):
        edges = self._G.get_edge_data(source, target)
        for key, attributes in edges.items():
            yield key, attributes

    def set_edge_data(self, source: str, target: str, edge_key: str, **kwargs) -> None:
        for key, value in kwargs.items():
            self._G[source][target][edge_key][key] = value

    def remove_edge(self, node_x: str, node_y: str, key: str) -> None:
        self._G.remove_edge(node_x, node_y, key)

    def get_edge_data(self, source: str, target: str, edge_key: str, key: str):
        return self._G[source][target][edge_key][key]

    def add_edge(
        self,
        source: str,
        target: str,
        code: str,
        edge_type: int,
        lineno: int=-1,
        col_offset: int=-1,
        end_lineno: int=-1,
        end_col_offset: int=-1
    ):
        if source and target and source != target and len(code) > 0:
            key = f"{source}_{target}_{code}"

            if self.has_edge(source, target, key=key):
                edge_count = self.get_edge_data(source, target, key, "count")
                self.set_edge_data(source, target, key, count=edge_count + 1)
            else:
                position = {
                    "lineno": lineno,
                    "col_offset": col_offset,
                    "end_lineno": end_lineno,
                    "end_col_offset": end_col_offset,
                }
                self._G.add_edge(source, target, code=code, key=key, edge_type=edge_type, count=1, **position)
        else:
            self._logger.debug("Ignoring edge %s -> %s with code %s", source, target, code)

    def optimize(self) -> None:
        edges_to_remove = []
        nodes_to_remove = []
        edges_to_add = []

        for node_x in self.node_iterator():
            node_degrees = self.node_degree(node_x)
            if node_degrees["in"] == 0 and node_degrees["out"] == 0:
                self._logger.debug("Removing node %s as it has no edges", node_x)
                nodes_to_remove.append(node_x)

            node_type_x = self.get_node(node_x).get("node_type")
            # reconnect and remove end_if nodes
            if node_type_x == NODE_TYPES["IF"]:
                optimize = True
                new_edges = []        
                for next_node in self.successor_node_iterator(node_x):
                    for prev_node in self.predecessor_node_iterator(node_x):                       
                        for _, attributes in self.get_edge_iterator(node_x, next_node):
                            if attributes["edge_type"] in [EDGE_TYPES["LOOP"], EDGE_TYPES["BRANCH"]]:
                                optimize = False
                            new_edges.append((prev_node, next_node, attributes["code"], attributes["edge_type"]))
                # we need the intermediate node for branches that follow directly after that node
                if optimize:
                    nodes_to_remove.append(node_x)
                    edges_to_add += new_edges
                # continue to the next node as we already did the optimization for this node and its edges
                continue

            for node_y in self.adjacent_node_iterator(node_x):
                edges_between_nodes = list(self.get_edge_iterator(node_x, node_y))

                if len(edges_between_nodes) > 1:
                    for key, edge_data in edges_between_nodes:
                        if (edge_data["code"] in node_x.lower() or edge_data["code"].replace(" ", "_") in node_x.lower()) and edge_data["edge_type"] == EDGE_TYPES["INPUT"]:
                            self._logger.debug("Removing edge %s -> %s with code %s as it is redundant", node_x, node_y, edge_data['code'])
                            edges_to_remove.append((node_x, node_y, key))
        # remove nodes and edges
        for node in nodes_to_remove:
            self.remove_node(node)

        for node_x, node_y, key in edges_to_remove:
            self.remove_edge(node_x, node_y, key)

        # add new edges
        for node_x, node_y, code, edge_type in edges_to_add:           
            self.add_edge(node_x, node_y, code, edge_type)
