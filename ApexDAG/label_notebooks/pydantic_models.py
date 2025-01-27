import re
import networkx as nx
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from textwrap import indent
from ApexDAG.sca.constants import REVERSE_NODE_TYPES, REVERSE_EDGE_TYPES
from ApexDAG.label_notebooks.message_template import DomainLabel


class Node(BaseModel):
    id: str
    node_type: str
    
    def __str__(self) -> str:
        return (f"Node(\n"
                f"  id={self.id},\n"
                f"  node_type={self.node_type}\n"
                f")")

    
class LabelledNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the node.")
    node_type: str = Field(..., description="Type of the node, e.g., VARIABLE, FUNCTION, etc.")
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
    ] = Field(..., description="Domain-specific label for the node.")
    class Config:
        use_enum_values = True  # Automatically serialize Enum values as strings for JSON

    def __str__(self) -> str:
        return (
            f"LabelledNode(\n"
            f"  id='{self.id}',\n"
            f"  node_type='{self.node_type}',\n"
            f"  domain_label='{self.domain_label}'\n"
            f")"
        )
    @classmethod
    def from_node(cls, node: Node, domain_label: DomainLabel) -> 'LabelledNode':
        return cls(id=node.id, node_type=node.node_type, domain_label=domain_label)

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


class GraphContext(BaseModel):
    nodes: List[Node | LabelledNode]
    edges: List[Edge]

    def get_neighbors(self, node_id: str):
        """Find neighbors (children and parents) of a given node."""
        children_edges = [edge for edge in self.edges if edge.source == node_id]
        parents_edges = [edge for edge in self.edges if edge.target == node_id]
        return children_edges, parents_edges


class GraphContextWithSubgraphSearch(GraphContext):
    def get_subgraph(self, node_id: str, max_depth: int = 1):
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

        dfs(node_id, 0)

        subgraph_nodes = list(subgraph_nodes)
        subgraph_nodes = [node for node in self.nodes if node.id in subgraph_nodes]
        
        return subgraph_nodes, subgraph_edges
    
    @classmethod
    def from_graph(cls, G: nx.DiGraph) -> 'GraphContextWithSubgraphSearch':

        for node in G.nodes:
            G.nodes[node]["label"] = re.sub(r"_\d+", "", node) # there are some very weird node names in the graph, so we need to clean them up a bit
            
        return cls(
            nodes=[Node(id=node_name, 
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
    node_of_interest: str

    def get_input_dict(self) -> Dict:
        """Prepare the input for the Groq API or model inference."""
        return self.model_dump()
    
    def __str__(self) -> str:
        # Use indentation to better format nodes and edges
        nodes_str = indent("\n".join([str(node) for node in self.nodes]), "  ")
        edges_str = indent("\n".join([str(edge) for edge in self.edges]), "  ")
        
        return (
            f"SubgraphContext(\n"
            f"  node_of_interest: {self.node_of_interest},\n"
            f"  nodes: [\n{nodes_str}\n  ],\n"
            f"  edges: [\n{edges_str}\n  ]\n"
            f")"
        )