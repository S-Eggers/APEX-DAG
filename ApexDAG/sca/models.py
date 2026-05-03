from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(kw_only=True)
class GraphNode:
    """
    Strict contract for all nodes in the ApexDAG ecosystem.
    """

    id: str | int
    label: str
    node_type: int
    cell_id: str
    code: str = ""
    lineno: int | None = None
    col_offset: int | None = None
    end_lineno: int | None = None
    end_col_offset: int | None = None

    predicted_label: int | None = None
    domain_label: str | None = None
    base_inputs: str | None = None
    transform_history: list[dict[str, Any]] = field(default_factory=list)

    def to_networkx_attrs(self) -> dict:
        raw_dict = asdict(self)
        raw_dict.pop("id", None)
        return {k: v for k, v in raw_dict.items() if v is not None}


@dataclass(kw_only=True)
class GraphEdge:
    """
    Strict contract for all edges in the ApexDAG ecosystem.
    """

    source: str | int
    target: str | int
    edge_type: int
    label: str = "edge"
    cell_id: str
    lineno: int | None = None
    col_offset: int | None = None
    end_lineno: int | None = None
    end_col_offset: int | None = None

    predicted_label: int | None = None
    domain_label: str | None = None

    def to_networkx_attrs(self) -> dict:

        raw_dict = asdict(self)
        raw_dict.pop("source", None)
        raw_dict.pop("target", None)
        return {k: v for k, v in raw_dict.items() if v is not None}


@dataclass
class ElementMetadata:
    name: str
    category: str
    label: str
    color: str
    border_style: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
