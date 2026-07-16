from pathlib import Path
from typing import Any, Protocol

import networkx as nx

from .domain import LineageTuple


class GraphParserPolicy(Protocol):
    def parse(self, raw_data: dict[str, Any]) -> nx.DiGraph: ...


class TupleExtractionPolicy(Protocol):
    def extract(self, graph: nx.DiGraph) -> list[LineageTuple]: ...


class TupleWriterPolicy(Protocol):
    def write(self, tuples: list[LineageTuple], output_path: Path) -> None: ...
