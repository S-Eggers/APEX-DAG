from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union

@dataclass(kw_only=True)
class GraphNode:
    """
    Strict contract for all nodes in the ApexDAG ecosystem.
    """
    id: Union[str, int]
    label: str
    node_type: int
    cell_id: str
    code: str = ""
    
    predicted_label: Optional[int] = None
    domain_label: Optional[str] = None
    base_inputs: Optional[str] = None
    transform_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_networkx_attrs(self) -> dict:
        raw_dict = asdict(self)
        raw_dict.pop("id", None)
        return {k: v for k, v in raw_dict.items() if v is not None}


@dataclass(kw_only=True)
class GraphEdge:
    """
    Strict contract for all edges in the ApexDAG ecosystem.
    """
    source: Union[str, int]
    target: Union[str, int]
    edge_type: int
    label: str = "edge"
    cell_id: str
    
    predicted_label: Optional[int] = None
    domain_label: Optional[str] = None

    def to_networkx_attrs(self) -> dict:

        raw_dict = asdict(self)
        raw_dict.pop("source", None)
        raw_dict.pop("target", None)
        return {k: v for k, v in raw_dict.items() if v is not None}