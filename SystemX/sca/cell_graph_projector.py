import ast
import builtins
import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum, auto

import networkx as nx

_BUILTIN_NAMES = frozenset(dir(builtins)) | {"display", "get_ipython", "__name__", "__file__", "_", "__", "___"}

_VERSION_SUFFIX_RE = re.compile(r"_[a-zA-Z0-9\-]{3,8}_\d+(?:_\d+)?$")

_FILE_WRITE_CALLS = frozenset({"to_csv", "to_parquet", "to_pickle", "to_excel", "to_json", "to_feather", "to_hdf", "savefig", "save", "dump", "save_model", "write_csv", "write_parquet", "torch_save"})
_FILE_READ_CALLS = frozenset({"read_csv", "read_parquet", "read_pickle", "read_excel", "read_json", "read_feather", "read_hdf", "load", "loadtxt", "load_model", "read_table", "scan_csv", "scan_parquet"})
_WRITE_MODES = frozenset({"w", "wb", "w+", "wb+", "a", "ab", "a+", "x", "xb"})

_MAGIC_OPAQUE_RE = re.compile(r"^[ \t]*(%%?(?:run|store|capture|bash|sh|script|writefile|load|px)|!)", re.MULTILINE)
_MAGIC_RUN_RE = re.compile(r"^[ \t]*%run\s+(\S+)", re.MULTILINE)
_SHELL_REDIRECT_RE = re.compile(r"^[ \t]*!.*?>>?\s*(\S+)", re.MULTILINE)

class DependencyKind(Enum):
    DATA = auto()
    IMPORT = auto()
    FUNCTION_DEF = auto()
    CLASS_DEF = auto()
    FILE_ARTIFACT = auto()
    OPAQUE = auto()

_DEF_KIND_TO_DEPENDENCY = {
    "import": DependencyKind.IMPORT,
    "function": DependencyKind.FUNCTION_DEF,
    "class": DependencyKind.CLASS_DEF,
    "assign": DependencyKind.DATA,
}

@dataclass
class CellDefinition:
    name: str
    kind: str
    lineno: int

@dataclass
class CellUse:
    name: str
    lineno: int
    deferred: bool = False
    via: str | None = None

@dataclass
class CellDependencyEdge:
    src_cell: str
    dst_cell: str
    kind: DependencyKind
    name: str
    def_node: str | None = None
    use_node: str | None = None
    def_lineno: int = -1
    use_lineno: int = -1
    ambiguous: bool = False
    candidate_def_cells: list[str] = field(default_factory=list)
    out_of_order: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "src_cell": self.src_cell,
            "dst_cell": self.dst_cell,
            "kind": self.kind.name,
            "name": self.name,
            "def_node": self.def_node,
            "use_node": self.use_node,
            "def_lineno": self.def_lineno,
            "use_lineno": self.use_lineno,
            "ambiguous": self.ambiguous,
            "candidate_def_cells": self.candidate_def_cells,
            "out_of_order": self.out_of_order,
            "confidence": self.confidence,
        }

class CellDependencyGraph:
    """Cell-level dependency multigraph plus order/constraint queries."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    @property
    def cells(self) -> list[str]:
        return sorted(self.graph.nodes, key=lambda c: self.graph.nodes[c]["document_index"])

    def edges(self) -> list[CellDependencyEdge]:
        return [data["edge"] for _, _, data in self.graph.edges(data=True)]

    def ambiguities(self) -> list[CellDependencyEdge]:
        return [e for e in self.edges() if e.ambiguous]

    def topological_order(self, tie_break: list[str] | None = None) -> list[str]:
        """Kahn's algorithm with a min-heap on the tie-break rank."""
        import heapq

        rank = {cell: i for i, cell in enumerate(tie_break or self.cells)}
        simple = nx.DiGraph(self.graph)
        indegree = dict(simple.in_degree())
        heap = [(rank[c], c) for c, d in indegree.items() if d == 0]
        heapq.heapify(heap)
        order: list[str] = []
        emitted: set[str] = set()

        while len(order) < len(indegree):
            if not heap:
                forced = min((c for c in indegree if c not in emitted), key=lambda c: rank[c])
                heapq.heappush(heap, (rank[forced], forced))
                indegree[forced] = 0
            _, cell = heapq.heappop(heap)
            if cell in emitted:
                continue
            emitted.add(cell)
            order.append(cell)
            for succ in simple.successors(cell):
                if succ in emitted:
                    continue
                indegree[succ] -= 1
                if indegree[succ] <= 0:
                    heapq.heappush(heap, (rank[succ], succ))
        return order

    def minimal_constraints(self) -> list[CellDependencyEdge]:
        """One representative edge per (src, dst) pair of the transitive reduction."""
        best: dict[tuple[str, str], CellDependencyEdge] = {}
        for edge in self.edges():
            key = (edge.src_cell, edge.dst_cell)
            if key not in best or edge.confidence > best[key].confidence:
                best[key] = edge

        simple = nx.DiGraph(best.keys())
        simple.add_nodes_from(self.graph.nodes)
        if nx.is_directed_acyclic_graph(simple):
            reduced = nx.transitive_reduction(simple)
            return [best[(u, v)] for u, v in reduced.edges]
        return list(best.values())

    def to_serializable(self) -> dict:
        nodes = []
        for cell, data in sorted(self.graph.nodes(data=True), key=lambda item: item[1]["document_index"]):
            nodes.append(
                {
                    "cell_id": cell,
                    "document_index": data["document_index"],
                    "execution_count": data.get("execution_count"),
                    "defined_names": sorted(data.get("defined_names", [])),
                    "free_uses": sorted(data.get("free_uses", [])),
                    "undefined_uses": sorted(data.get("undefined_uses", [])),
                    "has_opaque_effects": data.get("has_opaque_effects", False),
                    "is_dirty": data.get("is_dirty"),
                    "source_hash": data.get("source_hash", ""),
                    "source_preview": data.get("source_preview", ""),
                }
            )
        return {
            "cells": nodes,
            "edges": [e.to_dict() for e in self.edges()],
            "minimal_constraints": [e.to_dict() for e in self.minimal_constraints()],
            "ambiguities": [e.to_dict() for e in self.ambiguities()],
        }

class _ScopedUseCollector(ast.NodeVisitor):
    """Collect names read inside a function/class body that are not bound locally."""

    def __init__(self) -> None:
        self.free_reads: dict[str, int] = {}

    def collect(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef) -> dict[str, int]:
        bound: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                bound.add(arg.arg)
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
        body = node.body if isinstance(node.body, list) else [ast.Expr(value=node.body)]
        for stmt in body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                    bound.add(sub.id)
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        bound.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(sub, ast.arg):
                    bound.add(sub.arg)
        for stmt in body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in bound and sub.id not in _BUILTIN_NAMES:
                    self.free_reads.setdefault(sub.id, sub.lineno)
        return self.free_reads

class CellSymbolTable:
    """Per-cell module-scope definitions and free (externally-resolved) uses."""

    def __init__(self, source: str) -> None:
        self.defs: dict[str, CellDefinition] = {}
        self.uses: dict[str, CellUse] = {}
        self.parse_error = False
        try:
            tree = ast.parse(source)
        except SyntaxError:
            self.parse_error = True
            return
        self._bound: set[str] = set()
        self._deferred: dict[str, tuple[int, str | None]] = {}
        for stmt in tree.body:
            self._visit_statement(stmt)
        for name, (lineno, via) in self._deferred.items():
            if name not in self.defs and name not in self.uses:
                self.uses[name] = CellUse(name=name, lineno=lineno, deferred=True, via=via)

    def _define(self, name: str, kind: str, lineno: int) -> None:
        self._bound.add(name)
        self.defs.setdefault(name, CellDefinition(name=name, kind=kind, lineno=lineno))

    def _read(self, node: ast.Name) -> None:
        if node.id not in self._bound and node.id not in _BUILTIN_NAMES:
            self.uses.setdefault(node.id, CellUse(name=node.id, lineno=node.lineno))

    def _collect_reads(self, node: ast.AST | None) -> None:
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                self._read(sub)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                via = getattr(sub, "name", None)
                for name, lineno in _ScopedUseCollector().collect(sub).items():
                    self._deferred.setdefault(name, (lineno, via))
        for sub in ast.walk(node):
            if isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for gen in sub.generators:
                    for tgt in ast.walk(gen.target):
                        if isinstance(tgt, ast.Name):
                            self.uses.pop(tgt.id, None)

    def _bind_target(self, target: ast.AST, kind: str, lineno: int) -> None:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                self._define(sub.id, kind, lineno)
            elif isinstance(sub, (ast.Attribute, ast.Subscript)):
                self._collect_reads(sub.value if isinstance(sub, ast.Attribute) else sub)

    def _visit_statement(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            self._collect_reads(stmt.value)
            for target in stmt.targets:
                self._bind_target(target, "assign", stmt.lineno)
        elif isinstance(stmt, ast.AugAssign):
            self._collect_reads(stmt.value)
            self._collect_reads(stmt.target)
            self._bind_target(stmt.target, "assign", stmt.lineno)
        elif isinstance(stmt, ast.AnnAssign):
            self._collect_reads(stmt.value)
            if stmt.value is not None:
                self._bind_target(stmt.target, "assign", stmt.lineno)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                self._define((alias.asname or alias.name).split(".")[0], "import", stmt.lineno)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in stmt.decorator_list:
                self._collect_reads(deco)
            self._define(stmt.name, "function", stmt.lineno)
            for name, lineno in _ScopedUseCollector().collect(stmt).items():
                self._deferred.setdefault(name, (lineno, stmt.name))
        elif isinstance(stmt, ast.ClassDef):
            for deco in stmt.decorator_list:
                self._collect_reads(deco)
            for base in stmt.bases:
                self._collect_reads(base)
            self._define(stmt.name, "class", stmt.lineno)
            saved_defs = dict(self.defs)
            for sub in stmt.body:
                self._visit_statement(sub)
            self.defs = saved_defs
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._collect_reads(stmt.iter)
            self._bind_target(stmt.target, "assign", stmt.lineno)
            for sub in [*stmt.body, *stmt.orelse]:
                self._visit_statement(sub)
        elif isinstance(stmt, (ast.While, ast.If)):
            self._collect_reads(stmt.test)
            for sub in [*stmt.body, *stmt.orelse]:
                self._visit_statement(sub)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._collect_reads(item.context_expr)
                if item.optional_vars is not None:
                    self._bind_target(item.optional_vars, "assign", stmt.lineno)
            for sub in stmt.body:
                self._visit_statement(sub)
        elif isinstance(stmt, ast.Try):
            for sub in [*stmt.body, *stmt.orelse, *stmt.finalbody]:
                self._visit_statement(sub)
            for handler in stmt.handlers:
                self._collect_reads(handler.type)
                if handler.name:
                    self._define(handler.name, "assign", handler.lineno)
                for sub in handler.body:
                    self._visit_statement(sub)
        elif isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                self._bind_target(target, "assign", stmt.lineno)
        else:
            self._collect_reads(stmt)

def _extract_file_io(source: str) -> tuple[dict[str, int], dict[str, int]]:
    """Map literal file paths to first write / read lineno within one cell."""
    writes: dict[str, int] = {}
    reads: dict[str, int] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return writes, reads

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else None)
        if func_name is None:
            continue
        literal_args = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not literal_args:
            continue
        path = literal_args[0]
        if func_name == "open":
            mode = literal_args[1] if len(literal_args) > 1 else "r"
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            bucket = writes if mode in _WRITE_MODES or set(mode) & set("wax") else reads
            bucket.setdefault(path, node.lineno)
        elif func_name in _FILE_WRITE_CALLS:
            writes.setdefault(path, node.lineno)
        elif func_name in _FILE_READ_CALLS:
            reads.setdefault(path, node.lineno)
    return writes, reads

class CellGraphProjector:
    """Project a parsed DFG + raw cells onto a CellDependencyGraph."""

    def __init__(self, dfg, cells: list[dict], cache: dict | None = None) -> None:
        self._dfg_graph: nx.MultiDiGraph = dfg.get_graph() if hasattr(dfg, "get_graph") else dfg
        self._cells = [cell for cell in cells if str(cell.get("source", "")).strip()]
        self._cache = cache if cache is not None else {}

    def project(self) -> CellDependencyGraph:
        graph = nx.MultiDiGraph()
        cell_ids: list[str] = []
        tables: dict[str, CellSymbolTable] = {}

        table_cache = self._cache.setdefault("tables", {})
        for index, cell in enumerate(self._cells):
            cell_id = cell.get("cell_id") or cell.get("id") or f"cell_{index}"
            source = str(cell.get("source", ""))
            cell_ids.append(cell_id)
            table = table_cache.get(cell_id)
            if table is None:
                table = CellSymbolTable(self._sanitize(source))
                table_cache[cell_id] = table
            tables[cell_id] = table
            first_line = next((line for line in source.splitlines() if line.strip()), "")
            graph.add_node(
                cell_id,
                document_index=index,
                execution_count=cell.get("execution_count"),
                is_dirty=cell.get("is_dirty"),
                defined_names=set(table.defs),
                free_uses={u.name for u in table.uses.values()},
                undefined_uses=set(),
                has_opaque_effects=bool(_MAGIC_OPAQUE_RE.search(source)),
                source_hash=hashlib.sha1(source.encode()).hexdigest()[:12],
                source_preview=first_line[:120],
                n_lines=sum(1 for line in source.splitlines() if line.strip()),
                parse_error=table.parse_error,
            )

        dfg_index = self._index_dfg_cross_cell_edges(set(cell_ids))
        self._add_symbol_edges(graph, cell_ids, tables, dfg_index)
        self._add_file_artifact_edges(graph, cell_ids)
        self._add_magic_edges(graph, cell_ids)
        return CellDependencyGraph(graph)

    @staticmethod
    def _sanitize(source: str) -> str:
        source = re.sub(r"^[ \t]*[%!].*", "", source, flags=re.MULTILINE)
        return re.sub(r"^[ \t]*\?.*|.*\?[ \t]*$", "", source, flags=re.MULTILINE)

    def _index_dfg_cross_cell_edges(self, known_cells: set[str]) -> dict[tuple[str, str, str], dict]:
        """(src_cell, dst_cell, base_name) -> def/use DFG locations for the UI."""
        index: dict[tuple[str, str, str], dict] = {}
        nodes = self._dfg_graph.nodes
        for u, v, data in self._dfg_graph.edges(data=True):
            src_cell = nodes[u].get("cell_id")
            dst_cell = nodes[v].get("cell_id")
            if src_cell == dst_cell or src_cell not in known_cells or dst_cell not in known_cells:
                continue
            base_name = nodes[u].get("label") or _VERSION_SUFFIX_RE.sub("", str(u))
            key = (src_cell, dst_cell, base_name)
            if key not in index:
                index[key] = {
                    "def_node": u,
                    "use_node": v,
                    "def_lineno": self._node_lineno(u),
                    "use_lineno": int(data.get("lineno", -1)),
                    "edge_type": data.get("edge_type"),
                    "src_node_type": nodes[u].get("node_type"),
                }
        return index

    @staticmethod
    def _node_lineno(node_id: str) -> int:
        match = re.search(r"_(\d+)(?:_\d+)?$", str(node_id))
        return int(match.group(1)) if match else -1

    def _add_symbol_edges(self, graph: nx.MultiDiGraph, cell_ids: list[str], tables: dict[str, CellSymbolTable], dfg_index: dict) -> None:
        definers: dict[str, list[str]] = {}
        for cell_id in cell_ids:
            for name in tables[cell_id].defs:
                definers.setdefault(name, []).append(cell_id)

        for dst_cell in cell_ids:
            for use in tables[dst_cell].uses.values():
                if use.deferred:
                    self._add_deferred_use_edges(graph, cell_ids, tables, definers, dst_cell, use)
                    continue
                candidates = [c for c in definers.get(use.name, []) if c != dst_cell]
                if not candidates:
                    graph.nodes[dst_cell]["undefined_uses"].add(use.name)
                    continue
                self._emit_use_edge(graph, tables, dfg_index, dst_cell, use, candidates)

    def _emit_use_edge(
        self,
        graph: nx.MultiDiGraph,
        tables: dict[str, CellSymbolTable],
        dfg_index: dict,
        dst_cell: str,
        use: CellUse,
        candidates: list[str],
        confidence_cap: float = 1.0,
        allow_out_of_order: bool = True,
    ) -> None:
        if not candidates:
            return
        dst_index = graph.nodes[dst_cell]["document_index"]
        earlier = [c for c in candidates if graph.nodes[c]["document_index"] < dst_index]
        if earlier:
            src_cell = earlier[-1]
            out_of_order = False
        else:
            src_cell = min(candidates, key=lambda c: graph.nodes[c]["document_index"])
            out_of_order = allow_out_of_order
        if src_cell == dst_cell:
            return

        definition = tables[src_cell].defs[use.name]
        ambiguous = len(candidates) > 1
        located = dfg_index.get((src_cell, dst_cell, use.name), {})
        edge = CellDependencyEdge(
            src_cell=src_cell,
            dst_cell=dst_cell,
            kind=_DEF_KIND_TO_DEPENDENCY.get(definition.kind, DependencyKind.DATA),
            name=use.name,
            def_node=located.get("def_node"),
            use_node=located.get("use_node"),
            def_lineno=located.get("def_lineno", definition.lineno),
            use_lineno=located.get("use_lineno", use.lineno),
            ambiguous=ambiguous,
            candidate_def_cells=candidates,
            out_of_order=out_of_order,
            confidence=confidence_cap * ((1.0 / len(candidates)) if ambiguous else 1.0),
        )
        graph.add_edge(src_cell, dst_cell, edge=edge)

    def _add_deferred_use_edges(
        self,
        graph: nx.MultiDiGraph,
        cell_ids: list[str],
        tables: dict[str, CellSymbolTable],
        definers: dict[str, list[str]],
        def_cell: str,
        use: CellUse,
    ) -> None:
        """A read inside a function body resolves when the function is CALLED."""
        candidates = [c for c in definers.get(use.name, []) if c != def_cell]
        if not candidates:
            graph.nodes[def_cell]["undefined_uses"].add(use.name)
            return

        caller_cells = [c for c in cell_ids if use.via and c != def_cell and use.via in {u.name for u in tables[c].uses.values() if not u.deferred}]
        if caller_cells:
            for caller in caller_cells:
                caller_use = tables[caller].uses[use.via]
                call_site_use = CellUse(name=use.name, lineno=caller_use.lineno, deferred=True, via=use.via)
                self._emit_use_edge(graph, tables, {}, caller, call_site_use, [c for c in candidates if c != caller])
        else:
            self._emit_use_edge(graph, tables, {}, def_cell, use, candidates, confidence_cap=0.7, allow_out_of_order=False)

    def _add_file_artifact_edges(self, graph: nx.MultiDiGraph, cell_ids: list[str]) -> None:
        writers: dict[str, list[tuple[str, int]]] = {}
        readers: dict[str, list[tuple[str, int]]] = {}
        io_cache = self._cache.setdefault("file_io", {})
        for cell in self._cells:
            cell_id = cell.get("cell_id") or cell.get("id")
            if cell_id in io_cache:
                writes, reads = io_cache[cell_id]
            else:
                writes, reads = _extract_file_io(self._sanitize(str(cell.get("source", ""))))
                io_cache[cell_id] = (writes, reads)
            for path, lineno in writes.items():
                writers.setdefault(path, []).append((cell_id, lineno))
            for path, lineno in reads.items():
                readers.setdefault(path, []).append((cell_id, lineno))

        for path, reading in readers.items():
            for src_cell, def_lineno in writers.get(path, []):
                for dst_cell, use_lineno in reading:
                    if src_cell == dst_cell:
                        continue
                    candidates = [c for c, _ in writers[path]]
                    edge = CellDependencyEdge(
                        src_cell=src_cell,
                        dst_cell=dst_cell,
                        kind=DependencyKind.FILE_ARTIFACT,
                        name=path,
                        def_lineno=def_lineno,
                        use_lineno=use_lineno,
                        ambiguous=len(candidates) > 1,
                        candidate_def_cells=candidates,
                        confidence=0.6 / len(candidates),
                    )
                    graph.add_edge(src_cell, dst_cell, edge=edge)

    def _add_magic_edges(self, graph: nx.MultiDiGraph, cell_ids: list[str]) -> None:
        """Best-effort %run / shell-redirect artifact edges from raw source."""
        produces: dict[str, str] = {}
        for cell in self._cells:
            cell_id = cell.get("cell_id") or cell.get("id")
            source = str(cell.get("source", ""))
            for match in _SHELL_REDIRECT_RE.finditer(source):
                produces.setdefault(match.group(1), cell_id)
        for cell in self._cells:
            cell_id = cell.get("cell_id") or cell.get("id")
            source = str(cell.get("source", ""))
            for match in _MAGIC_RUN_RE.finditer(source):
                target = match.group(1)
                src_cell = produces.get(target)
                if src_cell and src_cell != cell_id:
                    edge = CellDependencyEdge(src_cell=src_cell, dst_cell=cell_id, kind=DependencyKind.OPAQUE, name=target, confidence=0.3)
                    graph.add_edge(src_cell, cell_id, edge=edge)
