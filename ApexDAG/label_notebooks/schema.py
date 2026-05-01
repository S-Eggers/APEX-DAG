import re
from enum import StrEnum
from textwrap import indent

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, FieldValidationInfo, field_validator

from ApexDAG.sca.constants import REVERSE_EDGE_TYPES, REVERSE_NODE_TYPES


# 1. Strict Alignment with New Taxonomy
class DomainLabel(StrEnum):
    """Strictly maps to DOMAIN_EDGE_TYPES constants."""

    MODEL_OPERATION = "Training, evaluating, tuning, or predicting with machine learning models."
    DATA_IMPORT_EXTRACTION = """Importing or extracting data from external sources
 (e.g., files, databases, APIs)."""
    DATA_TRANSFORM = """Transforming, cleaning, or preprocessing data
 (e.g., scaling, one-hot encoding, feature engineering)."""
    EDA = """Exploratory Data Analysis (e.g., plotting, printing stats,
 generating reports)."""
    DATA_EXPORT = "Exporting data to external storage, saving files."
    NOT_RELEVANT = """Not relevant to the core ML workflow
 (e.g., environment setup, logging, printing status, comments)."""

    @classmethod
    def get_prompt_description(cls) -> str:
        return "\n".join([f"- {label.name}: {label.value}" for label in cls])


class MultiNode(BaseModel):
    node_id: str
    node_type: str
    label: str

    def __str__(self) -> str:
        return f"""Node(id={self.node_id},
 node_type={self.node_type},
 label={self.label})"""


class MultiEdge(BaseModel):
    source: str
    target: str
    key: str
    edge_type: str
    code: str | None = None
    lineno: list[int] | None = None

    def __str__(self) -> str:
        return f"""Edge(source={self.source},
 target={self.target},
 key={self.key},
 code={self.code},
 edge_type={self.edge_type})
"""


class DomainKey(StrEnum):
    MODEL_OPERATION = "MODEL_OPERATION"
    DATA_IMPORT_EXTRACTION = "DATA_IMPORT_EXTRACTION"
    DATA_TRANSFORM = "DATA_TRANSFORM"
    EDA = "EDA"
    DATA_EXPORT = "DATA_EXPORT"
    NOT_RELEVANT = "NOT_RELEVANT"


# Mapping logic to keep truth in one place
DOMAIN_MAP = {
    DomainKey.MODEL_OPERATION: 0,
    DomainKey.DATA_IMPORT_EXTRACTION: 1,
    DomainKey.DATA_TRANSFORM: 2,
    DomainKey.EDA: 3,
    DomainKey.DATA_EXPORT: 4,
    DomainKey.NOT_RELEVANT: 5,
}


class MultiLabelledEdge(BaseModel):
    source: str
    target: str
    # ... other fields ...
    domain_label: str  # The LLM will provide the Key (e.g., "DATA_TRANSFORM")
    predicted_label: int = 0  # We will compute this
    reasoning: str

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("predicted_label", mode="before")
    @classmethod
    def map_label_to_id(cls, v: int, info: FieldValidationInfo) -> int:
        # Dynamically map the string key from the LLM to your DOMAIN_EDGES integer
        label_key = info.data.get("domain_label")
        return DOMAIN_MAP.get(label_key, 5)  # Default to NOT_RELEVANT


class MultiGraphContext(BaseModel):
    nodes: list[MultiNode]
    edges: list[MultiEdge | MultiLabelledEdge]

    @classmethod
    def from_graph(cls, graph: nx.MultiDiGraph) -> "MultiGraphContext":
        """Builds the strict DTO from a networkx.MultiDiGraph."""
        nodes = []
        for node_name, node_data in graph.nodes(data=True):
            clean_label = re.sub(r"_\d+", "", str(node_name))
            nodes.append(
                MultiNode(
                    node_id=str(node_name),
                    label=node_data.get("label", clean_label),
                    node_type=REVERSE_NODE_TYPES.get(node_data.get("node_type"), "UNKNOWN"),
                )
            )

        edges = []
        for src, tgt, key, edge_data in graph.edges(data=True, keys=True):
            lineno_start = edge_data.get("lineno")
            lineno_end = edge_data.get("end_lineno", lineno_start)
            lineno_range = list(range(lineno_start, lineno_end + 1)) if lineno_start is not None else None

            edges.append(
                MultiEdge(
                    source=str(src),
                    target=str(tgt),
                    key=str(key),
                    code=str(edge_data.get("code", "")),
                    edge_type=REVERSE_EDGE_TYPES.get(edge_data.get("edge_type"), "UNKNOWN"),
                    lineno=lineno_range,
                )
            )

        return cls(nodes=nodes, edges=edges)


class MultiSubgraphContext(MultiGraphContext):
    edge_of_interest: tuple[str, str, str]

    def __str__(self) -> str:
        nodes_str = indent("\n".join([str(node) for node in self.nodes]), "  ")
        edges_str = indent("\n".join([str(edge) for edge in self.edges]), "  ")
        return f"""SubgraphContext(
\n  edge_of_interest: {self.edge_of_interest},
\n  nodes: [\n{nodes_str}\n  ],
\n  edges: [\n{edges_str}\n  ]
\n)"""


class EdgeLabelPair(BaseModel):
    edge_id: str = Field(..., description="The unique ID/key of the edge being labeled.")
    label: MultiLabelledEdge


class BatchLabelResponse(BaseModel):
    labels: list[EdgeLabelPair]
