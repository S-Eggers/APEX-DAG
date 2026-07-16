import networkx as nx

from SystemX.execution.diagnostics import derive_cell_states
from SystemX.execution.types import ExecutionStateReport
from SystemX.sca.cell_graph_projector import CellDependencyGraph

_OPAQUE_ANCESTOR_DISCOUNT = 0.8

class HeuristicOrderPredictor:
    def predict(self, cg: CellDependencyGraph, dfg=None, tie_break: list[str] | None = None) -> ExecutionStateReport:
        """dfg is accepted for signature parity with the learned predictor (which pools node features from it) and ignored here."""
        unknown_cells, dropped_edges = self._break_cycles(cg)
        order = cg.topological_order(tie_break=tie_break)
        rank = {cell: i for i, cell in enumerate(order)}

        states, flags = derive_cell_states(cg, unknown_cells)
        flags["dropped_cycle_edges"] = [e.to_dict() for e in dropped_edges]

        simple = nx.DiGraph(cg.graph)
        opaque_cells = {cell for cell, data in cg.graph.nodes(data=True) if data.get("has_opaque_effects")}
        for cell in cg.graph.nodes:
            confidence = 1.0
            for _, _, data in cg.graph.in_edges(cell, data=True):
                confidence *= data["edge"].confidence
            ancestors = nx.ancestors(simple, cell)
            if ancestors & opaque_cells or cell in opaque_cells:
                confidence *= _OPAQUE_ANCESTOR_DISCOUNT
            states[cell].confidence = confidence
            states[cell].predicted_rank = rank[cell]

        return ExecutionStateReport(
            predicted_order=order,
            constraints=[e.to_dict() for e in cg.minimal_constraints()],
            ambiguities=[e.to_dict() for e in cg.ambiguities()],
            cell_states=states,
            notebook_flags=flags,
        )

    @staticmethod
    def _break_cycles(cg: CellDependencyGraph) -> tuple[set[str], list]:
        """Drop the lowest-confidence edge of each remaining cycle (mutates cg)."""
        unknown_cells: set[str] = set()
        dropped = []
        while True:
            simple = nx.DiGraph(cg.graph)
            try:
                cycle = nx.find_cycle(simple)
            except nx.NetworkXNoCycle:
                return unknown_cells, dropped

            weakest_pair = None
            weakest_conf = float("inf")
            for u, v in ((u, v) for u, v, *_ in cycle):
                pair_conf = max(data["edge"].confidence for data in cg.graph.get_edge_data(u, v).values())
                if pair_conf < weakest_conf:
                    weakest_conf = pair_conf
                    weakest_pair = (u, v)

            u, v = weakest_pair
            for key, data in list(cg.graph.get_edge_data(u, v).items()):
                dropped.append(data["edge"])
                cg.graph.remove_edge(u, v, key)
            unknown_cells.update((u, v))
