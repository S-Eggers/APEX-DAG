import warnings

import networkx as nx

from ApexDAG.sca.constants import NODE_TYPES
from ApexDAG.sca.graph_utils import (
    convert_multidigraph_to_digraph,
    load_graph,
    save_graph,
)


class LegacyIOMixin:
    """
    Quarantine class for deprecated I/O, Graph Visualization, and Pass-through methods.
    """

    def _warn_deprecated(self, method_name: str) -> None:
        warnings.warn(
            f"'{method_name}' is deprecated and will be removed in a future release. Retrieve the state via 'get_state()' and orchestrate externally.",
            DeprecationWarning,
            stacklevel=3,
        )

    def set_domain_label(self, attrs: dict, name: str) -> None:
        self._warn_deprecated("set_domain_label")
        nx.set_edge_attributes(self._current_state._G, attrs, name=name)

    def set_domain_node_label(self, attrs: dict, name: str) -> None:
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
