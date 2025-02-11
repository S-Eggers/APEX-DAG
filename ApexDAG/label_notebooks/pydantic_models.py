import re
import networkx as nx
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from textwrap import indent
from ApexDAG.sca.constants import REVERSE_NODE_TYPES, REVERSE_EDGE_TYPES
from ApexDAG.label_notebooks.message_template import DomainLabel


class Node(BaseModel):
    node_id: str
    node_type: str
    
    def __str__(self) -> str:
        return (f"Node(\n"
                f"  id={self.node_id},\n"
                f"  node_type={self.node_type}\n"
                f")")


class Edge(BaseModel):
    source: str  
    target: str 
    edge_type: str
    code: Optional[str] = None
    
    def __str__(self) -> str:
        return (f"Edge(\n"
               f"  source={self.source},\n" 
               f"  target={self.target},\n"
               f"  code={self.code},\n"
               f"  edge_type={self.edge_type}\n"
               f")")

    
class LabelledEdge(BaseModel):
    '''
    This class is used to serialize the edge with the edge domain label
    '''
    source: str = Field(..., description="Unique identifier for the source of the edge.")
    target: str = Field(..., description="Unique identifier for the target of the edge.")
    code: str = Field(str, description="The code that connects the source and target nodes.")
    edge_type: str = Field(..., description="Type of the edge")
    domain_label: Literal[ # the literal is very important Enum does not work as well 
        "MODEL_TRAIN", 
        "MODEL_EVALUATION", 
        "HYPERPARAMETER_TUNING", 
        "DATA_EXPORT", 
        "DATA_IMPORT_EXTRACTION", 
        "DATA_TRANSFORM", 
        "EDA", 
        "ENVIRONMENT", 
        "NOT_INTERESTING"
    ] = Field(..., description="Domain-specific label for the edge.")
    
    class Config:
        title = "LabelledEdge"
        description = (
            "This class is used to serialize the edge with the domain label. "
            "It ensures that the `source`, `target` and `node_type` are strings and that the `domain_label` "
            "is one of the specified literals."
        )
        use_enum_values = True  # Automatically serialize Enum values as strings for JSON

    def __str__(self) -> str:
        return (
            f"LabelledEdge(\n"
            f"  source='{self.source}',\n"
            f"  target='{self.target}',\n"
            f"  code='{self.code}',\n"
            f"  edge_type='{self.edge_type}',\n"
            f"  domain_label='{self.domain_label}'\n"
            f")"
        )
    @classmethod
    def from_edge(cls, node: Edge, domain_label: DomainLabel) -> 'LabelledEdge':
        return cls(source=node.source,
                   target=node.target,
                   edge_type=node.edge_type, 
                   code=node.code,
                   domain_label=domain_label)


class GraphContext(BaseModel):
    nodes: List[Node]
    edges: List[Edge | LabelledEdge]
    
    edge_dict: Dict[str, List[str]] = Field(default_factory=dict)

    def get_neighbors(self, node_id: str):
        """Find neighbors (children and parents) of a given node."""
        children_edges = [edge for edge in self.edges if edge.source == node_id]
        parents_edges = [edge for edge in self.edges if edge.target == node_id]
        return children_edges, parents_edges
    
    def populate_edge_dict(self): 
        """Populate the edge_dict with sources as keys and lists of target indices as values."""
        self.edge_dict.clear()
        for index, edge in enumerate(self.edges):
            if edge.source not in self.edge_dict:
                self.edge_dict[edge.source] = {}
            self.edge_dict[edge.source][edge.target] = index


class GraphContextWithSubgraphSearch(GraphContext):
    def get_subgraph(self, node_id_source: str, node_id_target: str, max_depth: int = 1):
        """Extract subgraph focusing on the node, its parents, and grandparents."""
        visited = set()
        subgraph_nodes = set()
        subgraph_edges = []

        def dfs(current_node_id, current_depth):
            if current_depth >= max_depth or current_node_id in visited:
                visited.add(current_node_id)
                subgraph_nodes.add(current_node_id)
                return
            subgraph_nodes.add(current_node_id)
            visited.add(current_node_id)

            children_edges, parents_edges = self.get_neighbors(current_node_id)
            for child_edge in children_edges:
                if child_edge.target not in visited:
                    subgraph_edges.append(child_edge) # do ot rather with an id..... 
                    dfs(child_edge.target, current_depth + 1)
            for parent_edge in parents_edges:
                if parent_edge.source not in visited:
                    subgraph_edges.append(parent_edge)
                    dfs(parent_edge.source, current_depth + 1)

        dfs(node_id_source, 0)
        dfs(node_id_target, 0)

        subgraph_nodes = list(subgraph_nodes)
        subgraph_nodes = [node for node in self.nodes if node.node_id in subgraph_nodes]
        
        return subgraph_nodes, subgraph_edges
    
    @classmethod
    def from_graph(cls, G: nx.DiGraph) -> 'GraphContextWithSubgraphSearch':

        for node in G.nodes:
            G.nodes[node]["label"] = re.sub(r"_\d+", "", node) # there are some very weird node names in the graph, so we need to clean them up a bit
            
        return cls(
            nodes=[Node(node_id=node_name, 
                        label = node_object["label"], 
                        node_type=REVERSE_NODE_TYPES[node_object["node_type"]]
                        ) for node_name, node_object in G.nodes._nodes.items()],
            edges=[Edge(source=source, 
                        target=target, 
                        code=G.edges._adjdict[source][target]["code"], 
                        edge_type=REVERSE_EDGE_TYPES[G.edges._adjdict[source][target]["edge_type"]]
                        ) for (source, target) in G.edges]
        )


class SubgraphContext(GraphContext):
    edge_of_interest: tuple[str, str]

    def get_input_dict(self) -> Dict:
        """Prepare the input for the Groq API or model inference."""
        return self.model_dump()
    
    def __str__(self) -> str:

        nodes_str = indent("\n".join([str(node) for node in self.nodes]), "  ")
        edges_str = indent("\n".join([str(edge) for edge in self.edges]), "  ")
        
        return (
            f"SubgraphContext(\n"
            f"  edge_of_interest: {self.edge_of_interest},\n"
            f"  nodes: [\n{nodes_str}\n  ],\n"
            f"  edges: [\n{edges_str}\n  ]\n"
            f")"
        )