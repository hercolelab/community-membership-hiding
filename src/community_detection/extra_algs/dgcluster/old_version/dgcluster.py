import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
#from karateclub import Node2Vec
import igraph as ig
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from src.utils.utils import DRL_agentHyps, DatasetNames
from sklearn.cluster import Birch
import numpy as np
import cdlib
import logging
import time
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec
from torch.optim import SparseAdam
from torch_geometric.data import Data
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../../..'))


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, base_model):
        super(GNN, self).__init__()

        if base_model == 'gcn':
            self.conv1 = GCNConv(in_dim, 256)
            self.conv2 = GCNConv(256, 128)
            self.conv3 = GCNConv(128, out_dim)
        elif base_model == 'gat':
            self.conv1 = GATConv(in_dim, 256)
            self.conv2 = GATConv(256, 128)
            self.conv3 = GATConv(128, out_dim)
        elif base_model == 'gin':
            self.conv1 = GINConv(nn.Linear(in_dim, 256))
            self.conv2 = GINConv(nn.Linear(256, 128))
            self.conv3 = GINConv(nn.Linear(128, out_dim))
        elif base_model == 'sage':
            self.conv1 = SAGEConv(in_dim, 256)
            self.conv2 = SAGEConv(256, 128)
            self.conv3 = SAGEConv(128, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        x = x / (x.sum())
        x = (F.tanh(x)) ** 2
        x = F.normalize(x)

        return x
    

def from_ig_graph_to_node2vec_geometric_data(graph: ig.Graph) -> Data:
    """
    Convert an igraph graph to a PyTorch Geometric Data object.
    """
    
    # Salva il livello di log corrente
    original_level = logging.root.level
    
    # Imposta temporaneamente il livello di log a ERROR per silenziare i messaggi di info/warning
    logging.root.setLevel(logging.ERROR)
    
    try:
        g_undir = graph.as_undirected()
        nx_graph = g_undir.to_networkx()
        data = from_networkx(nx_graph)
        edge_index = data.edge_index

        # hyperparameters
        embedding_dim = 128
        walk_length   = 20
        context_size  = 10
        walks_per_node= 10
        p, q          = 1.0, 1.0      # return / in-out parameters
        num_negative_samples = 1
        sparse = True                # use SparseAdam optimizer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = Node2Vec(
            edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            num_negative_samples=num_negative_samples,
            num_nodes=graph.vcount(),
            sparse=sparse
        ).to(device)

        loader = model.loader(batch_size=128, shuffle=True)
        optimizer = SparseAdam(list(model.parameters()), lr=0.01)

        def train_epoch():
            model.train()
            total_loss = 0

            for pos_rw, neg_rw in loader:
                pos_rw = pos_rw.to(device)
                neg_rw = neg_rw.to(device)

                optimizer.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            return total_loss

        # Run multiple epochs
        for epoch in range(1, 11):
            loss = train_epoch()
            #print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

        model.eval()
        with torch.no_grad():
            # shape [num_nodes, embedding_dim]
            embeddings = model.embedding.weight.data.cpu()
        
        data = Data(x=embeddings, edge_index=edge_index)
        return data

    finally:
        # Ripristina il livello di log originale
        logging.root.setLevel(original_level)

def DGCluster(graph: ig.Graph, graph_name: str) -> list:
    """
    DGCluster algorithm for community detection.
    
    Parameters
    ----------
    graph : ig.Graph
        The graph to be clustered
    graph_name : str
        The name of the graph
    
    Returns
    -------
    list
        List of clusters
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_name = getattr(DatasetNames, graph_name).value
    model_path = f"{ROOT_DIR}/src/community_detection/extra_algs/dgcluster/models/model_{graph_name}_0.2_301_gcn_22.pth"
    # Convert the igraph graph to PyTorch Geometric Data object
    data = from_ig_graph_to_node2vec_geometric_data(graph)
    data = data.to(device)
    # Load the pre-trained model
    in_dim = data.x.shape[1]
    out_dim = 64
    model = GNN(in_dim, out_dim, base_model='gcn').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    x = model(data)
    clusters = Birch(n_clusters=None, threshold=0.5).fit_predict(x.detach().cpu().numpy(), y=None)
    if clusters.max() < 0:
        # If there aren't any clusters, create a single cluster
        dgclusters = [list(range(len(clusters)+1))]
    else:
        dgclusters = [list(np.where(clusters == i)[0]) for i in range(clusters.min(), clusters.max() + 1)]
    node_cluster = cdlib.NodeClustering(dgclusters, graph, method_name="DGCluster")
    
    return node_cluster