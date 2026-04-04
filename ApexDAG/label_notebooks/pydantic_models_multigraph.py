import re
import networkx as nx
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Union
from textwrap import indent
from ApexDAG.sca.constants import REVERSE_NODE_TYPES, REVERSE_EDGE_TYPES
from ApexDAG.label_notebooks.message_template import DomainLabel


class MultiNode(BaseModel):
    """Represents a node in a MultiDiGraph."""
    node_id: str
    node_type: str
    label: str

    def __str__(self) -> str:
        return f"Node(id={self.node_id}, node_type={self.node_type}, label={self.label})"


class MultiEdge(BaseModel):
    """Represents an edge in a MultiDiGraph."""
    source: str
    target: str
    key: str
    edge_type: str
    code: Optional[str] = None
    lineno: Optional[List[int]] = None

    def __str__(self) -> str:
        return f"Edge(source={self.source}, target={self.target}, key={self.key}, code={self.code}, edge_type={self.edge_type})"


class MultiLabelledEdge(BaseModel):
    """Represents a labelled edge in a MultiDiGraph, including a domain label."""
    source: str = Field(
        ..., description="Unique identifier for the source of the edge."
    )
    target: str = Field(
        ..., description="Unique identifier for the target of the edge."
    )
    key: str = Field(
        ..., description="Unique key for the edge between source and target."
    )
    code: Optional[str] = Field(
        None, description="The code that connects the source and target nodes."
    )
    edge_type: str = Field(..., description="Type of the edge")
    lineno: Optional[List[int]] = Field(
        None,
        description="The line number of the code that connects the source and target nodes.",
    )
    domain_label: Literal[
        "MODEL_TRAIN",
        "MODEL_EVALUATION",
        "HYPERPARAMETER_TUNING",
        "DATA_EXPORT",
        "DATA_IMPORT_EXTRACTION",
        "DATA_TRANSFORM",
        "EDA",
        "ENVIRONMENT",
        "NOT_INTERESTING",
        "MISSING",
        "MORE_CONTEXT_NEEDED",
    ] = Field(..., description="Domain-specific label for the edge.")
    reasoning: Optional[str] = Field(
        None, description="The reasoning provided by the LLM for the domain label."
    )

    class Config:
        title = "MultiLabelledEdge"
        description = (
            "This class is used to serialize a multigraph edge with its domain label."
        )
        use_enum_values = True

    def __str__(self) -> str:
        return f"LabelledEdge(source='{self.source}', target='{self.target}', key='{self.key}', code='{self.code}', edge_type='{self.edge_type}', domain_label='{self.domain_label}', reasoning='{self.reasoning}')"

    @classmethod
    def from_edge(
        cls, edge: MultiEdge, domain_label: DomainLabel, reasoning: Optional[str] = None
    ) -> "MultiLabelledEdge":
        """Creates a MultiLabelledEdge from a MultiEdge and a domain label."""
        return cls(
            source=edge.source,
            target=edge.target,
            key=edge.key,
            edge_type=edge.edge_type,
            code=edge.code,
            lineno=edge.lineno,
            domain_label=domain_label,
            reasoning=reasoning,
        )


class MultiGraphContext(BaseModel):
    """Represents the context of a full MultiDiGraph."""
    nodes: List[MultiNode]
    edges: List[Union[MultiEdge, MultiLabelledEdge]]

    edge_dict: Dict[str, Dict[str, Dict[int, int]]] = Field(default_factory=dict)

    def get_neighbors(self, node_id: str):
        """Finds neighbors (children and parents) of a given node."""
        children_edges = [edge for edge in self.edges if edge.source == node_id]
        parents_edges = [edge for edge in self.edges if edge.target == node_id]
        return children_edges, parents_edges

    def populate_edge_dict(self):
        """Populates the edge_dict: source -> target -> key -> edge_index."""
        self.edge_dict.clear()
        for index, edge in enumerate(self.edges):
            self.edge_dict.setdefault(edge.source, {}).setdefault(
                edge.target, {}
            )[edge.key] = index


class MultiGraphContextWithSubgraphSearch(MultiGraphContext):
    """Extends MultiGraphContext with subgraph extraction capabilities."""

    def get_subgraph(
        self,
        node_id_source: str,
        node_id_target: str,
        max_depth: int = 1,
    ):
        """
        Extracts a subgraph by finding all nodes within max_depth from both
        source and target nodes, and then including all edges between those nodes.
        """
        visited = set()
        subgraph_node_ids = set()

        def dfs(current_node_id, current_depth):
            if current_depth > max_depth or current_node_id in visited:
                return
            
            subgraph_node_ids.add(current_node_id)
            visited.add(current_node_id)

            children_edges, parents_edges = self.get_neighbors(current_node_id)
            for child_edge in children_edges:
                dfs(child_edge.target, current_depth + 1)
            for parent_edge in parents_edges:
                dfs(parent_edge.source, current_depth + 1)

        # Run DFS from both start points to collect all relevant nodes
        dfs(node_id_source, 0)
        dfs(node_id_target, 0)

        # Filter the full node and edge lists to create the subgraph context
        subgraph_nodes = [
            node for node in self.nodes if node.node_id in subgraph_node_ids
        ]
        subgraph_edges = [
            edge
            for edge in self.edges
            if edge.source in subgraph_node_ids and edge.target in subgraph_node_ids
        ]

        return subgraph_nodes, subgraph_edges

    @classmethod
    def from_graph(
        cls,
        G: nx.MultiDiGraph
    ) -> "MultiGraphContextWithSubgraphSearch":
        """Builds the context from a networkx.MultiDiGraph."""
        for node_name in G.nodes:
            G.nodes[node_name]["label"] = re.sub(r"_\d+", "", node_name)

        def build_node(node_name: str, node_data: dict) -> MultiNode:
            return MultiNode(
                node_id=node_name,
                label=node_data.get("label", ""),
                node_type=REVERSE_NODE_TYPES.get(
                    node_data.get("node_type"), "UNKNOWN"
                ),
            )
        
        def build_edge(
            source: str, target: str, key: int, edge_data: dict
        ) -> MultiEdge:
            lineno_start: Optional[int] = edge_data.get("lineno")
            lineno_end: Optional[int] = edge_data.get("end_lineno", lineno_start)
            lineno_range = (
                list(range(lineno_start, lineno_end + 1))
                if lineno_start is not None
                else None
            )

            return MultiEdge(
                source=source,
                target=target,
                key=str(key),
                code=edge_data.get("code", ""),
                edge_type=REVERSE_EDGE_TYPES.get(
                    edge_data.get("edge_type"), "UNKNOWN"
                ),
                lineno=lineno_range,
            )

        nodes = [build_node(name, data) for name, data in G.nodes(data=True)]
        edges = [
            build_edge(src, tgt, key, data)
            for src, tgt, key, data in G.edges(data=True, keys=True)
        ]
        
        instance = cls(nodes=nodes, edges=edges)
        instance.populate_edge_dict()
        return instance


class MultiSubgraphContext(MultiGraphContext):
    """Represents the context of a specific subgraph for analysis."""
    edge_of_interest: tuple[str, str, str]

    def get_input_dict(self) -> Dict:
        """Prepares the input for a model inference."""
        return self.model_dump()

    def __str__(self) -> str:
        nodes_str = indent("\n".join([str(node) for node in self.nodes]), "  ")
        edges_str = indent("\n".join([str(edge) for edge in self.edges]), "  ")

        return f"MultiSubgraphContext(edge_of_interest: {self.edge_of_interest}, nodes: [{nodes_str}], edges: [{edges_str}])"