from typing import TypedDict


class GoldenElementData(TypedDict, total=False):
    """Strict schema for the 'data' payload of a Golden Graph element."""

    id: str
    label: str
    node_type: int
    cell_id: str
    code: str
    transform_history: list[str]
    source: str
    target: str
    edge_type: int
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    count: int
    raw_code: str
    domain_label: str
    predicted_label: int


class GoldenElement(TypedDict):
    """Strict schema for a Golden Graph element wrapper."""

    data: GoldenElementData


class GoldenGraph(TypedDict):
    """Strict schema for the root Golden Graph JSON."""

    elements: list[GoldenElement]
