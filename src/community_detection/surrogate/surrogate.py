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
from sklearn.cluster import DBSCAN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../../..'))


class SurrogateGNN(nn.Module):
    def __init__(self, n_nodes, emb_dim=128, hidden1=256, hidden2=256, out_dim=64, num_proj=128):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, emb_dim)
        self.gcn1 = GCNConv(emb_dim, hidden1)
        self.gcn2 = GCNConv(hidden1, hidden2)
        self.gcn3 = GCNConv(hidden2, out_dim) 
        self.clf_head = nn.Linear(out_dim, num_proj)  # logits for clustering

    def forward(self, data):
        x = self.embedding(torch.arange(data.num_nodes, device=data.edge_index.device))
        x = self.gcn1(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn3(x, data.edge_index)
        z = F.normalize(x, p=2, dim=-1)  # clustering embedding
        logits = self.clf_head(z)        # for CE loss
        return z, logits  
    

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
        data = Data(edge_index=edge_index)
        data.num_nodes = graph.vcount()
        return data

    finally:
        # Ripristina il livello di log originale
        logging.root.setLevel(original_level)

def SurrogateCluster(graph: ig.Graph, graph_name: str) -> list:
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
    #graph_name = getattr(DatasetNames, graph_name).value
    model_path = f"{ROOT_DIR}/temp_cmh/src/community_detection/surrogate/models/surrogate_gcn_{graph_name}_100_0.001_22.pth"
    #model_path = f"{ROOT_DIR}/temp_cmh/src/community_detection/surrogate/models/surrogate_gcn_{graph_name}.pth"
    # Convert the igraph graph to PyTorch Geometric Data object
    data = from_ig_graph_to_node2vec_geometric_data(graph)
    data = data.to(device)
    # Load the pre-trained model
    in_dim = 128
    out_dim = 64
    model = SurrogateGNN(n_nodes=data.num_nodes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    z, logits = model(data)
    emb_np = z.detach().cpu().numpy()
    db = DBSCAN(eps=0.5, min_samples=3, metric='euclidean')  # You can tune eps
    pred = db.fit_predict(emb_np)
    if pred.max() < 0:
        # If there aren't any clusters, create a single cluster
        surrogate = [list(range(len(pred)+1))]
    else:
        surrogate = [list(np.where(pred == i)[0]) for i in range(pred.min(), pred.max() + 1)]
    node_cluster = cdlib.NodeClustering(surrogate, graph, method_name="DGCluster")
    
    return node_cluster