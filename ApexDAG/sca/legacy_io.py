import warnings

import networkx as nx

from ApexDAG.sca.constants import EDGE_TYPES, NODE_TYPES
from ApexDAG.sca.graph_utils import (
    convert_multidigraph_to_digraph,
    get_subgraph,
    load_graph,
    save_graph,
)
from ApexDAG.util.draw import Draw


class LegacyIOMixin:
    """
    Quarantine class for deprecated I/O, Graph Visualization, and Pass-through methods.
    """

    def _warn_deprecated(self, method_name: str):
        warnings.warn(
            f"'{method_name}' is deprecated and will be removed in a future release. "
            "Retrieve the state via 'get_state()' and orchestrate externally.",
            DeprecationWarning,
            stacklevel=3
        )

    def draw_all_subgraphs(self) -> None:
        self._warn_deprecated("draw_all_subgraphs")
        for variable in self._current_state.variable_versions:
            self.draw(variable, variable)
        self.draw()

    def draw(self, save_path: str = None, start_node: str = None) -> None:
        self._warn_deprecated("draw")
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
            draw.dfg(G, save_path)

    def webrender(self, save_path: str = None) -> None:
        self._warn_deprecated("webrender")
        draw = Draw(NODE_TYPES, EDGE_TYPES)
        G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
        draw.dfg_webrendering(G, save_path)

    def set_domain_label(self, attrs: dict, name: str):
        self._warn_deprecated("set_domain_label")
        nx.set_edge_attributes(self._current_state._G, attrs, name=name)

    def set_domain_node_label(self, attrs: dict, name: str):
        self._warn_deprecated("set_domain_node_label")
        nx.set_node_attributes(self._current_state._G, attrs, name=name)

    def save_dfg(self, path: str) -> None:
        self._warn_deprecated("save_dfg")
        G = convert_multidigraph_to_digraph(self._current_state.get_graph(), NODE_TYPES)
        save_graph(G, path)

    def read_dfg(self, path: str) -> None:
        self._warn_deprecated("read_dfg")
        self._current_state.set_graph(load_graph(path))
        if hasattr(self, "_logger"):
            self._logger.info("Graph successfully loaded from %s", path)

    def optimize(self) -> None:
        self._warn_deprecated("optimize")
        self._current_state.optimize()

    def filter_relevant(self, lineage_mode: bool = False) -> None:
        self._warn_deprecated("filter_relevant")
        self._current_state.filter_relevant(lineage_mode=lineage_mode)
