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
    def __init__(self, n_nodes, n_clusters, emb_dim=128, hidden1=256, hidden2=256):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, emb_dim)
        self.gcn1 = GCNConv(emb_dim, hidden1)
        self.gcn2 = GCNConv(hidden1, hidden2)
        self.gcn3 = GCNConv(hidden2, n_clusters)  # Output: logits per cluster

    def forward(self, data):
        x = self.embedding(torch.arange(data.num_nodes, device=data.edge_index.device))
        x = self.gcn1(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn3(x, data.edge_index)
        return x  # shape: [n_nodes, n_clusters]

def train_surrogate_gnn(dataset, n_epochs=100, lr=1e-3, device='cpu'):
    all_graphs = [d['perturbed_graph'] for d in dataset]
    n_nodes = all_graphs[0].vcount()
    n_clusters = max([np.max(d['agg_clusters']) for d in dataset]) + 1
    model = SurrogateGNN(n_nodes, n_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(n_epochs):
        total_loss = 0
        model.train()
        for entry in dataset:
            pert_graph = entry['perturbed_graph']
            target_labels = np.array(entry['agg_clusters'])  # shape: (n_nodes,)
            data = from_ig_graph_to_geometric_data(pert_graph)
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)  # shape: [n_nodes, n_clusters]
            target_tensor = torch.tensor(target_labels, dtype=torch.long, device=device)
            loss = criterion(logits, target_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
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
    model = train_surrogate_gnn(surrogate_dataset, n_epochs=100, lr=1e-3, device=device)
    torch.save(model.state_dict(), f'src/community_detection/surrogate/models/surrogate_gcn_{dataset_name}.pth')
    print("Training completed and model saved.")

if __name__ == "__main__":
    main() 