import yaml
from dataclasses import dataclass
from ApexDAG.label_notebooks.message_template import DomainLabel
from ApexDAG.label_notebooks.pydantic_models import GraphContextWithSubgraphSearch, SubgraphContext

@dataclass
class Config:
    model_name: str
    sleep_interval: int
    
def get_input_subgraph(graph_context: GraphContextWithSubgraphSearch, node_id: str, max_depth: int = 1) -> DomainLabel:
    subgraph_nodes, subgraph_edges = graph_context.get_subgraph(node_id, max_depth = max_depth)
    model_input = SubgraphContext(
        node_of_interest=node_id,
        nodes=subgraph_nodes,
        edges=subgraph_edges
    )
    input_graph_structure = str(model_input)
    return input_graph_structure 


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)