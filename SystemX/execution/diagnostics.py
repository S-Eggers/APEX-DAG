import networkx as nx

from SystemX.execution.types import CellState, CellStatus
from SystemX.sca.cell_graph_projector import CellDependencyGraph

def derive_cell_states(cg: CellDependencyGraph, unknown_cells: set[str]) -> tuple[dict[str, CellState], dict]:
    graph = cg.graph
    states = {cell: CellState(cell_id=cell, status=CellStatus.FRESH) for cell in graph.nodes}
    flags: dict = {"safe_to_run_top_to_bottom": True, "hidden_state_risk": False, "out_of_order_evidence": False}

    doc_index = {cell: data["document_index"] for cell, data in graph.nodes(data=True)}
    counts = {cell: data.get("execution_count") for cell, data in graph.nodes(data=True)}

    for cell in unknown_cells:
        states[cell].status = CellStatus.UNKNOWN
        states[cell].reasons.append("part of a dependency cycle that had to be broken")

    for cell, data in graph.nodes(data=True):
        if data.get("parse_error"):
            states[cell].status = CellStatus.UNKNOWN
            states[cell].reasons.append("cell does not parse; analysis is incomplete")
        if data.get("undefined_uses"):
            names = ", ".join(sorted(data["undefined_uses"]))
            states[cell].hidden_state_risk = True
            states[cell].reasons.append(f"reads name(s) no current cell defines: {names} (deleted cell or magic residue)")
            flags["hidden_state_risk"] = True
        if counts[cell] is None and states[cell].status is CellStatus.FRESH:
            states[cell].status = CellStatus.UNEXECUTED
            states[cell].reasons.append("never executed in this kernel session")

    stale_seeds: set[str] = set()
    for edge in cg.edges():
        src, dst = edge.src_cell, edge.dst_cell

        if edge.out_of_order:
            flags["safe_to_run_top_to_bottom"] = False
            flags["out_of_order_evidence"] = True
        if doc_index[src] > doc_index[dst]:
            states[dst].unsafe_reorder = True
            states[dst].reasons.append(f"depends on `{edge.name}` from a cell further down the notebook")
            flags["safe_to_run_top_to_bottom"] = False

        src_count, dst_count = counts[src], counts[dst]
        if src_count is not None and dst_count is not None and src_count > dst_count:
            stale_seeds.add(dst)
            states[dst].reasons.append(f"`{edge.name}` was recomputed (upstream cell ran after this one)")
        if dst_count is not None and src_count is None:
            if states[dst].status in (CellStatus.FRESH, CellStatus.STALE):
                states[dst].status = CellStatus.MISSING_DEPS
            states[dst].reasons.append(f"was executed, but its dependency for `{edge.name}` never ran in this session")

    for cell, data in graph.nodes(data=True):
        if data.get("is_dirty") and counts[cell] is not None:
            stale_seeds.add(cell)
            if states[cell].status is CellStatus.FRESH:
                states[cell].status = CellStatus.DIRTY
            states[cell].reasons.append("source edited after its last execution")

    simple = nx.DiGraph(graph)
    poisoned: set[str] = set()
    for seed in stale_seeds:
        poisoned.add(seed)
        poisoned.update(nx.descendants(simple, seed))
    for cell in poisoned:
        if states[cell].status is CellStatus.FRESH:
            states[cell].status = CellStatus.STALE
            if cell not in stale_seeds:
                states[cell].reasons.append("downstream of a stale or edited cell")

    return states, flags
