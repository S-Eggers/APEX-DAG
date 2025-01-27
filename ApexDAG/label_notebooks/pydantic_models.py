from pydantic import BaseModel
from typing import List, Optional, Dict
import re
import networkx as nx
from ApexDAG.sca.constants import REVERSE_NODE_TYPES, REVERSE_EDGE_TYPES
from ApexDAG.label_notebooks.message_template import DomainLabel


class Node(BaseModel):
    id: str
    node_type: str
    
class LabelledNode(BaseModel):
    id: str
    node_type: str
    domain_label: DomainLabel
    
    @classmethod
    def from_node(cls, node: Node, domainlabel: DomainLabel) -> 'LabelledNode':
        return cls(id=node.id, node_type=node.node_type, domainlabel=domainlabel)

class Edge(BaseModel):
    source: str  
    target: str 
    edge_type: str
    code: Optional[str] = None

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