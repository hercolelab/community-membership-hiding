import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from karateclub import Node2Vec
import igraph as ig
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from src.utils.utils import DRL_agentHyps, DatasetNames
from sklearn.cluster import Birch
import numpy as np
import cdlib
import logging


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
        nx_graph = graph.to_networkx()
        
        embedding_model = Node2Vec(
            walk_number=DRL_agentHyps.WALK_NUMBER.value,
            walk_length=DRL_agentHyps.WALK_LENGTH.value,
            dimensions=DRL_agentHyps.EMBEDDING_DIM.value,
        )
        embedding_model.fit(nx_graph)
        embedding = embedding_model.get_embedding()
        
        # Crea manualmente le strutture dati necessarie
        num_nodes = len(nx_graph.nodes())
        x = torch.zeros((num_nodes, DRL_agentHyps.EMBEDDING_DIM.value))
        for node in nx_graph.nodes():
            x[node] = torch.tensor(embedding[node])
        
        # Crea la lista degli archi
        edge_index = []
        for u, v in nx_graph.edges():
            edge_index.append([u, v])
            edge_index.append([v, u])  # Aggiungi arco in entrambe le direzioni
        
        edge_index = torch.tensor(edge_index).t()
        
        # Crea l'oggetto Data
        data = Data(x=x, edge_index=edge_index)
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
    model_path = f"src/community_detection/extra_algs/dgcluster/dgc_models/model_{graph_name}_02_500_gcn_22.pth"
    # Convert the igraph graph to PyTorch Geometric Data object
    data = from_ig_graph_to_node2vec_geometric_data(graph)
    data = data.to(device)
    # Load the pre-trained model
    in_dim = data.x.shape[1]
    out_dim = 64
    model = GNN(in_dim, out_dim, base_model='gcn').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    # Perform inference
    x = model(data)
    clusters = Birch(n_clusters=None, threshold=0.5).fit_predict(x.detach().cpu().numpy(), y=None)
    dgclusters = [list(np.where(clusters == i)[0]) for i in range(clusters.max() + 1)]
    node_cluster = cdlib.NodeClustering(dgclusters, graph, method_name="DGCluster")
    
    return node_cluster