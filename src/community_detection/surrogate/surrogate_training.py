import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from src.community_detection.surrogate.surrogate_dataset import generate_surrogate_dataset
from src.graph_environment.env import GraphEnvironment
from src.utils.utils import Utils, FilePaths
import igraph as ig
from torch_geometric.utils import from_networkx
from scipy.spatial.distance import squareform
from sklearn.cluster import Birch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import random

def from_ig_graph_to_geometric_data(graph: ig.Graph) -> Data:
    """
    Convert an igraph graph to a PyTorch Geometric Data object.
    """
    
    nx_graph = graph.to_networkx()
    nx_graph = nx_graph.to_undirected()
    data = from_networkx(nx_graph)
    edge_index = data.edge_index 
    data = Data(edge_index=edge_index)
    data.num_nodes = graph.vcount()
    return data

class SurrogateGNN(nn.Module):
    def __init__(self, n_nodes, emb_dim=128, hidden1=256, hidden2=256, out_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, emb_dim)
        self.gcn1 = GCNConv(emb_dim, hidden1)
        self.gcn2 = GCNConv(hidden1, hidden2)
        self.gcn3 = GCNConv(hidden2, out_dim) 

    def forward(self, data):
        x = self.embedding(torch.arange(data.num_nodes, device=data.edge_index.device))
        x = self.gcn1(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn3(x, data.edge_index)
        return x  

def hungarian_match(pred_labels, target_labels, num_clusters):
    contingency = np.zeros((num_clusters, num_clusters), dtype=np.int32)
    for p, t in zip(pred_labels, target_labels):
        if t == -1: continue
        contingency[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    match = {r: c for r, c in zip(row_ind, col_ind)}
    remapped = [match.get(p, -1) for p in pred_labels]
    return torch.tensor(remapped, dtype=torch.long)

def train_surrogate_gnn(dataset, n_epochs=100, lr=1e-3, device='cpu'):
    n_nodes = dataset[0]['perturbed_graph'].vcount()
    model = SurrogateGNN(n_nodes=n_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for entry in dataset:
            data = from_ig_graph_to_geometric_data(entry['perturbed_graph']).to(device)
            target = np.array(entry['agg_clusters'])  # shape (n_nodes,)

            optimizer.zero_grad()
            logits = model(data)  # [n_nodes, emb_dim]
            emb = logits.detach().cpu().numpy()

            # Clustering agnostico sui logit
            birch = Birch(threshold=0.5, n_clusters=None)
            pred = birch.fit_predict(emb)

            n_clusters_target = len(set(target)) - (1 if -1 in target else 0)
            pred_matched = hungarian_match(pred, target, n_clusters_target).to(device)

            loss = criterion(logits, pred_matched)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / len(dataset):.4f}")

    return model


def main():
    dataset_name = "KAR"
    env = GraphEnvironment(dataset_name,["LEID"])
    graph = env.original_graph.copy()

    surrogate_dataset = generate_surrogate_dataset(graph, env)
    if not surrogate_dataset:
        print("Dataset empty! Check graph and env.")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed = 22
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    num_epochs = 10
    learning_rate = 1e-3
    model = train_surrogate_gnn(surrogate_dataset, n_epochs=num_epochs, lr=learning_rate, device=device)
    torch.save(model.state_dict(), f'src/community_detection/surrogate/models/surrogate_gcn_{dataset_name}_{num_epochs}_{learning_rate}_{seed}.pth')
    print("Training completed and model saved.")

if __name__ == "__main__":
    main() 