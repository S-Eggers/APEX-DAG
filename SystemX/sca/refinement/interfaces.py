from __future__ import annotations

from typing import Protocol

import networkx as nx

from .state import RefinementState


class GraphRefinementRule(Protocol):
    """Protocol for all graph refinement heuristics."""

    def apply(self, g: nx.MultiDiGraph, state: RefinementState) -> None: ...
