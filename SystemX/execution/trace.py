from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from SystemX.sca.cell_graph_projector import CellDependencyGraph, CellSymbolTable

TRACE_SCHEMA_VERSION = 1

@dataclass
class TraceEvent:
    seq: int
    cell_id: str
    document_index: int
    execution_count: int | None
    source_hash: str
    timestamp: str
    session_id: str
    success: bool | None = None

@dataclass
class TraceSession:
    session_id: str
    kernel_id: str | None
    started_at: str
    ended_reason: str | None
    events: list[TraceEvent] = field(default_factory=list)

def djb2_hash(source: str) -> str:
    """Bit-exact port of the frontend's hashSource (djb2 over UTF-16 code units, wrapped to signed 32-bit, rendered in base 36 with a sign)."""
    raw = source.encode("utf-16-le")
    value = 5381
    for i in range(0, len(raw), 2):
        code = raw[i] | (raw[i + 1] << 8)
        value = (value * 33 + code) & 0xFFFFFFFF
    if value >= 0x80000000:
        signed = value - 0x100000000
    else:
        signed = value
    if signed < 0:
        return "-" + _to_base36(-signed)
    return _to_base36(signed)

def _to_base36(value: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    out = []
    while value:
        value, rem = divmod(value, 36)
        out.append(digits[rem])
    return "".join(reversed(out))

def parse_trace(payload: dict | None) -> tuple[list[TraceSession], dict[str, str]]:
    """Tolerant loader for the request/sidecar trace dict."""
    if not isinstance(payload, dict):
        return [], {}

    sources = payload.get("sources")
    sources = {str(k): str(v) for k, v in sources.items()} if isinstance(sources, dict) else {}

    sessions: list[TraceSession] = []
    for raw in payload.get("sessions") or []:
        if not isinstance(raw, dict):
            continue
        session_id = str(raw.get("session_id") or f"session_{len(sessions)}")
        events: list[TraceEvent] = []
        for raw_event in raw.get("events") or []:
            if not isinstance(raw_event, dict) or "cell_id" not in raw_event:
                continue
            count = raw_event.get("execution_count")
            events.append(
                TraceEvent(
                    seq=int(raw_event.get("seq", len(events))),
                    cell_id=str(raw_event["cell_id"]),
                    document_index=int(raw_event.get("document_index", -1)),
                    execution_count=int(count) if isinstance(count, int) else None,
                    source_hash=str(raw_event.get("source_hash", "")),
                    timestamp=str(raw_event.get("timestamp", "")),
                    session_id=session_id,
                    success=raw_event.get("success") if isinstance(raw_event.get("success"), bool) else None,
                )
            )
        events.sort(key=lambda e: e.seq)
        sessions.append(
            TraceSession(
                session_id=session_id,
                kernel_id=str(raw["kernel_id"]) if raw.get("kernel_id") is not None else None,
                started_at=str(raw.get("started_at", "")),
                ended_reason=str(raw["ended_reason"]) if raw.get("ended_reason") is not None else None,
                events=events,
            )
        )
    return sessions, sources

def observed_order(session: TraceSession) -> list[str]:
    """Cell ids ordered by their *last* execution in the session."""
    last_seq: dict[str, int] = {}
    for event in session.events:
        last_seq[event.cell_id] = event.seq
    return sorted(last_seq, key=lambda c: last_seq[c])

def kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    """Pair-concordance tau over the items common to both orders (no scipy)."""
    common = [item for item in order_a if item in set(order_b)]
    if len(common) < 2:
        return None
    rank_b = {item: i for i, item in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            if rank_b[common[i]] < rank_b[common[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)

def _last_seq_map(session: TraceSession) -> dict[str, int]:
    return {cell: seq for cell, seq in ((e.cell_id, e.seq) for e in session.events)}

def analyze_session(
    cg: CellDependencyGraph,
    session: TraceSession,
    current_cell_ids: set[str],
    sources: dict[str, str],
    predicted_order: list[str],
) -> dict:
    """Replay one session's events over the dependency graph."""
    last_seq = _last_seq_map(session)
    order = observed_order(session)

    in_edges_by_dst: dict[str, list] = {}
    for edge in cg.edges():
        in_edges_by_dst.setdefault(edge.dst_cell, []).append(edge)

    annotated_events: list[dict] = []
    hash_history: dict[str, str] = {}
    executed_so_far: set[str] = set()
    max_doc_index = -1
    for event in session.events:
        previous_hash = hash_history.get(event.cell_id)
        if previous_hash is None:
            rerun_kind = "first"
        elif previous_hash == event.source_hash:
            rerun_kind = "identical"
        else:
            rerun_kind = "edited"
        hash_history[event.cell_id] = event.source_hash

        stale_input_names = sorted(
            {
                edge.name
                for edge in in_edges_by_dst.get(event.cell_id, [])
                if edge.src_cell not in executed_so_far
            }
        )
        out_of_doc_order = 0 <= event.document_index < max_doc_index
        max_doc_index = max(max_doc_index, event.document_index)
        executed_so_far.add(event.cell_id)

        annotated_events.append(
            {
                "seq": event.seq,
                "cell_id": event.cell_id,
                "document_index": event.document_index,
                "execution_count": event.execution_count,
                "timestamp": event.timestamp,
                "success": event.success,
                "in_notebook": event.cell_id in current_cell_ids,
                "rerun_kind": rerun_kind,
                "out_of_doc_order": out_of_doc_order,
                "stale_input_names": stale_input_names,
            }
        )

    violated: list[dict] = []
    for edge in cg.edges():
        src_seq = last_seq.get(edge.src_cell)
        dst_seq = last_seq.get(edge.dst_cell)
        if dst_seq is None:
            continue
        if src_seq is None:
            violated.append({**edge.to_dict(), "violation": "missing-upstream"})
        elif src_seq > dst_seq:
            violated.append({**edge.to_dict(), "violation": "reversed"})

    residue_consumers: set[str] = set()
    for cell in cg.graph.nodes:
        residue_consumers.update(cg.graph.nodes[cell].get("undefined_uses", ()))
    deleted_cell_reads: list[dict] = []
    seen_deleted: set[str] = set()
    for event in session.events:
        if event.cell_id in current_cell_ids or event.cell_id in seen_deleted:
            continue
        seen_deleted.add(event.cell_id)
        deleted_source = sources.get(event.source_hash)
        defined: set[str] = set()
        if deleted_source is not None:
            defined = set(CellSymbolTable(deleted_source).defs)
        deleted_cell_reads.append(
            {
                "cell_id": event.cell_id,
                "names": sorted(defined & residue_consumers),
                "defined_names": sorted(defined),
                "deleted_source_hash": event.source_hash if deleted_source is not None else None,
            }
        )

    document_order = cg.cells
    return {
        "session_id": session.session_id,
        "kernel_id": session.kernel_id,
        "started_at": session.started_at,
        "ended_reason": session.ended_reason,
        "observed_order": order,
        "is_linear_extension": not violated,
        "violated_constraints": violated,
        "kendall_tau_vs_predicted": kendall_tau(order, predicted_order),
        "kendall_tau_vs_document": kendall_tau(order, document_order),
        "deleted_cell_reads": deleted_cell_reads,
        "events": annotated_events,
    }

def freshness_map(
    cg: CellDependencyGraph,
    session: TraceSession | None,
    cells: list[dict],
) -> dict[str, str]:
    """fresh | edited | unexecuted | stale-input per current cell."""
    last_hash: dict[str, str] = {}
    if session is not None:
        for event in session.events:
            last_hash[event.cell_id] = event.source_hash

    current_hash = {
        str(cell.get("cell_id") or cell.get("id")): djb2_hash(str(cell.get("source", "")))
        for cell in cells
    }

    freshness: dict[str, str] = {}
    for cell in cg.graph.nodes:
        if cell not in last_hash:
            freshness[cell] = "unexecuted"
        elif cg.graph.nodes[cell].get("is_dirty") or last_hash[cell] != current_hash.get(cell):
            freshness[cell] = "edited"
        else:
            freshness[cell] = "fresh"

    simple = nx.DiGraph(cg.graph)
    for cell in cg.graph.nodes:
        if freshness[cell] == "fresh" and any(
            freshness.get(ancestor) != "fresh" for ancestor in nx.ancestors(simple, cell)
        ):
            freshness[cell] = "stale-input"
    return freshness

def minimal_replay_set(
    cg: CellDependencyGraph,
    freshness: dict[str, str],
    target: str,
) -> list[str]:
    """Cells to re-run so target's inputs and output are fresh, in a valid execution order: every non-fresh ancestor plus the target itself when it is not fresh."""
    if target not in cg.graph:
        return []
    simple = nx.DiGraph(cg.graph)
    needed = {cell for cell in nx.ancestors(simple, target) if freshness.get(cell) != "fresh"}
    if freshness.get(target) != "fresh":
        needed.add(target)
    return [cell for cell in cg.topological_order() if cell in needed]

def reproducibility_report(
    cg: CellDependencyGraph,
    analysis: dict | None,
    freshness: dict[str, str],
) -> dict:
    """Would "Restart & Run All" (document order, current sources) reproduce the analyzed session's final state?"""
    risks: list[dict] = []

    for edge in cg.edges():
        if edge.out_of_order:
            risks.append(
                {
                    "kind": "out-of-order-binding",
                    "cell_id": edge.dst_cell,
                    "detail": f"`{edge.name}` is only defined further down the notebook; top-to-bottom execution cannot rebuild it",
                }
            )

    executed: set[str] = set()
    if analysis is not None:
        executed = set(analysis["observed_order"])
        for read in analysis["deleted_cell_reads"]:
            if read["names"]:
                names = ", ".join(read["names"])
                risks.append(
                    {
                        "kind": "deleted-cell-state",
                        "cell_id": read["cell_id"],
                        "detail": f"a deleted cell defined {names}, still consumed by current cells",
                    }
                )

        last_seq = {cell: i for i, cell in enumerate(analysis["observed_order"])}
        seen_pairs: set[tuple[str, str]] = set()
        for edge in cg.ambiguities():
            key = (edge.dst_cell, edge.name)
            if key in seen_pairs or edge.dst_cell not in last_seq:
                continue
            seen_pairs.add(key)
            candidates_run = [
                c for c in edge.candidate_def_cells if c in last_seq and last_seq[c] < last_seq[edge.dst_cell]
            ]
            if candidates_run:
                exercised = max(candidates_run, key=lambda c: last_seq[c])
                if exercised != edge.src_cell:
                    risks.append(
                        {
                            "kind": "ambiguous-binding-divergence",
                            "cell_id": edge.dst_cell,
                            "detail": f"`{edge.name}` was last written by a different cell than document order would pick",
                        }
                    )

    for cell in cg.graph.nodes:
        state = freshness.get(cell)
        if state == "edited" and cell in executed:
            risks.append(
                {
                    "kind": "edited-after-run",
                    "cell_id": cell,
                    "detail": "source edited after its last execution; a rerun executes different code than produced the current state",
                }
            )
        if state == "unexecuted" and any(
            edge.dst_cell in executed for edge in cg.edges() if edge.src_cell == cell
        ):
            risks.append(
                {
                    "kind": "never-executed",
                    "cell_id": cell,
                    "detail": "defines state consumed by executed cells but never ran this session; a full rerun would rebind it",
                }
            )
        if cg.graph.nodes[cell].get("has_opaque_effects") and cell in executed:
            risks.append(
                {
                    "kind": "opaque-effects",
                    "cell_id": cell,
                    "detail": "contains magics or shell escapes; replay is best-effort",
                }
            )

    return {"run_all_reproduces": not risks, "risks": risks}
