import torch
import networkx as nx

class DegreeBasedHeuristic:
    def __init__(self, device: torch.device):
        self.device = device
        self.prob_model_start = torch.tensor([0, 0.4, 0.3, 0, 0.3], device=self.device)
        self.prob_model_end = torch.tensor([0.25, 0.15, 0.20, 0.15, 0.25], device=self.device)

    def apply(self, probabilities: torch.Tensor, nx_G: nx.DiGraph, graph_edges_list: list) -> torch.Tensor:
        in_degrees = nx_G.in_degree()
        out_degrees = nx_G.out_degree()
        
        start_mask = torch.tensor(
            [in_degrees[u] == 0 for u, v, k, d in graph_edges_list],
            dtype=torch.bool,
            device=self.device
        )
        end_mask = torch.tensor(
            [out_degrees[v] == 0 for u, v, k, d in graph_edges_list],
            dtype=torch.bool,
            device=self.device
        )
        
        end_mask &= ~start_mask
        
        probabilities[start_mask] *= self.prob_model_start
        probabilities[end_mask] *= self.prob_model_end
        
        return probabilities