import torch
import networkx as nx
from torch.nn import Embedding
from torch_geometric.nn.models import Node2Vec
from torch_geometric.utils import from_networkx


def write_dict_to_file(file_path, dictionary):
    with open(file_path, "w") as f:
        for key, value in dictionary.items():
            f.write(f"{key}: {value}\n")
            
def load_dict_from_file(file_path):
    dictionary = dict()
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.split(":")
            dictionary[key] = value.strip()
    return dictionary

def split_code(code: str):
    return code.split()

def networkx_to_pyc(g: nx.Graph):
    word2idx = dict()
    for node in g.nodes:
        label = g.nodes[node]["label"]
        code = code = g.nodes[node]["code"]
        content = [label] + split_code(code)
        for word in content:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    
    node_embeddings = Embedding(len(word2idx), 256)
    for node in g.nodes:
        label = g.nodes[node]["label"]
        # ignore code for now
        # code = g.nodes[node]["code"]
        embedding = node_embeddings(torch.tensor(word2idx[label]))
        g.nodes[node]["embedding"] = embedding
        attributes = list(g.nodes[node].keys())
        for attr in attributes:
            if attr not in ["embedding"]:
                del g.nodes[node][attr]
                
    write_dict_to_file("output/word2idx.txt", word2idx)
    return from_networkx(g, group_node_attrs=["embedding"])

def node2vec_embedding(data):
    # ToDo: Implement method
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=256, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(1, 10):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss / len(loader))
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask], z[data.test_mask], data.y[data.test_mask], max_iter=10)
    
    emb = model()
    return emb