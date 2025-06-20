import numpy as np
import random
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from sklearn.cluster import Birch
import networkx as nx
import argparse
import os
import json
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec
from torch.optim import SparseAdam
from torch_geometric.data import Data
from enum import Enum
import igraph as ig
from torch_geometric.data import Data
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../../..'))

class FilePaths(Enum):
    """Class to store file paths for data and models"""

    # Local
    DATASETS_DIR = f"{ROOT_DIR}/dataset/networks"
    LOG_DIR = "src/logs/"
    TEST_DIR = "outputs/"

    # Used Datasets
    KAR = DATASETS_DIR + "/kar.txt"
    WORDS = DATASETS_DIR + "/words.txt"
    VOTE = DATASETS_DIR + "/vote.txt"
    POW = DATASETS_DIR + "/pow.txt"
    FB_75 = DATASETS_DIR + "/fb-75.txt"
    COND_MAT = DATASETS_DIR + "/cond-mat.txt"
    FB_ART = DATASETS_DIR + "/fb-artist.txt"
    DBLP = DATASETS_DIR + "/dblp.txt"
    YT = DATASETS_DIR + "/youtube.txt"

def import_graph(file_path: str) -> ig.Graph:
        """
        Import a graph from a txt file using igraph, ensure nodes are labeled from 0 to n-1,
        and store original labels as vertex 'name' attribute.

        Parameters
        ----------
        file_path : str
            File path of the .txt file
            
        Returns
        -------
        ig.Graph
            Graph with consecutive integer node labels and original names preserved
        """
        if not file_path.endswith(".txt"):
            raise ValueError("File format not supported")
        
        # Load raw edge list
        with open(file_path, "r") as f:
            edges = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
        # Extract all unique nodes
        unique_nodes = sorted(set([node for edge in edges for node in edge]))
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
        # Relabel edges with new node ids
        relabeled_edges = [(node_mapping[src], node_mapping[dst]) for src, dst in edges]
        # Create graph from relabeled edge list
        graph = ig.Graph(edges=relabeled_edges, directed=False)
        #graph = graph.simplify(multiple=True, loops=True) # avoided because we apply pre_process_graph
        # Store original node IDs as 'name' attribute
        graph.vs["name"] = unique_nodes 
        # Return the biggest connected component    # avoided because we apply pre_process_graph
        #largest_component = graph.clusters().giant()

        return graph



def compute_fast_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree, device):
    mx = max(clusters)
    MM = np.zeros((num_nodes, mx + 1))
    for i in range(len(clusters)):
        MM[i][clusters[i]] = 1
    MM = torch.tensor(MM).double().to(device)

    x = torch.matmul(torch.t(MM), torch_sparse_adj.double())
    x = torch.matmul(x, MM)
    x = torch.trace(x)

    y = torch.matmul(torch.t(MM), degree.double())
    y = torch.matmul(torch.t(y.unsqueeze(dim=0)), y.unsqueeze(dim=0))
    y = torch.trace(y)
    y = y / (2 * num_edges)
    return ((x - y) / (2 * num_edges)).item()


def parse_args():
    args = argparse.ArgumentParser(description='DGCluster arguments.')
    args.add_argument('--dataset', type=str, default='kar', choices=['kar', 'words', 'vote', 'pow', 'fb-75', 'cond-mat', 'fb-art', 'dblp'])
    args.add_argument('--lam', type=float, default=0.2)
    args.add_argument('--alp', type=float, default=0)
    args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0'])
    args.add_argument('--epochs', type=int, default=300)
    args.add_argument('--base_model', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
    args.add_argument('--seed', type=int, default=22)
    args = args.parse_args()
    return args

def from_ig_graph_to_node2vec_geometric_data(graph: ig.Graph) -> Data:
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
    

def load_dataset(dataset_name):
    if dataset_name == 'kar':
        graph = import_graph(file_path=FilePaths.KAR.value)
        dataset = from_ig_graph_to_node2vec_geometric_data(graph)
    elif dataset_name == 'words':
        graph = import_graph(file_path=FilePaths.WORDS.value)
        dataset = from_ig_graph_to_node2vec_geometric_data(graph)
    elif dataset_name == 'vote':
        graph = import_graph(file_path=FilePaths.VOTE.value)
        dataset = from_ig_graph_to_node2vec_geometric_data(graph)
    elif dataset_name == 'pow':
        graph = import_graph(file_path=FilePaths.POW.value)
        dataset = from_ig_graph_to_node2vec_geometric_data(graph)
    elif dataset_name == 'fb-75':
        graph = import_graph(file_path=FilePaths.FB_75.value)
        dataset = from_ig_graph_to_node2vec_geometric_data(graph)
    elif dataset_name == 'cond-mat':
        graph = import_graph(file_path=FilePaths.COND_MAT.value)
        dataset = from_ig_graph_to_node2vec_geometric_data(graph)
    elif dataset_name == 'fb-art':
        graph = import_graph(file_path=FilePaths.FB_ART.value)
        dataset = from_ig_graph_to_node2vec_geometric_data(graph)
    elif dataset_name == 'dblp':
        graph = import_graph(file_path=FilePaths.DBLP.value)
        dataset = from_ig_graph_to_node2vec_geometric_data(graph)
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} not implemented.')
    return dataset

class GNN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, base_model):
        super(GNN, self).__init__()

        if base_model == 'gcn':
            self.embed = nn.Embedding(num_nodes, in_dim)
            self.conv1 = GCNConv(in_dim, 256)
            self.conv2 = GCNConv(256, 128)
            self.conv3 = GCNConv(128, out_dim)
        elif base_model == 'gat':
            self.embed = nn.Embedding(num_nodes, in_dim)
            self.conv1 = GATConv(in_dim, 256)
            self.conv2 = GATConv(256, 128)
            self.conv3 = GATConv(128, out_dim)
        elif base_model == 'gin':
            self.embed = nn.Embedding(num_nodes, in_dim)
            self.conv1 = GINConv(nn.Linear(in_dim, 256))
            self.conv2 = GINConv(nn.Linear(256, 128))
            self.conv3 = GINConv(nn.Linear(128, out_dim))
        elif base_model == 'sage':
            self.embed = nn.Embedding(num_nodes, in_dim)
            self.conv1 = SAGEConv(in_dim, 256)
            self.conv2 = SAGEConv(256, 128)
            self.conv3 = SAGEConv(128, out_dim)

    def forward(self, data):
        #x, edge_index = data.x, data.edge_index
        x = self.embed(torch.arange(data.num_nodes, device=data.edge_index.device))
        edge_index = data.edge_index

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

def convert_scipy_torch_sp(sp_adj):
    sp_adj = sp_adj.tocoo()
    indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
    sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
    return sp_adj


def regularization(output, s):
    out = output[s, :]
    ss = out.sum(dim=0)
    ss = ss ** 2
    ss = ss.sum()
    avg_sim = 1 / (len(s) ** 2) * ss

    return avg_sim ** 2

def loss_fn(output, lam=0.0, alp=0.0, epoch=-1):
    sample_size = int(1 * num_nodes)
    s = random.sample(range(0, num_nodes), sample_size)

    s_output = output[s, :]

    s_adj = sparse_adj[s, :][:, s]
    s_adj = convert_scipy_torch_sp(s_adj)
    s_degree = degree[s]

    x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(device))
    x = torch.matmul(x, s_output.double())
    x = torch.trace(x)

    y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(device))
    y = (y ** 2).sum()
    y = y / (2 * num_edges)

    # scaling=1
    scaling = num_nodes ** 2 / (sample_size ** 2)

    m_loss = -((x - y) / (2 * num_edges)) * scaling

    reg_loss = alp * regularization(output, s)

    loss = m_loss + reg_loss

    print('epoch: ', epoch, 'loss: ', loss.item(), 'm_loss: ', m_loss.item(), 'reg_loss: ', reg_loss.item())

    return loss

def train(model, optimizer, data, epochs, lam, alp):
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)

        loss = loss_fn(out, lam, alp, epoch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    lam = args.lam
    alp = args.alp
    epochs = args.epochs
    device = args.device
    base_model = args.base_model
    seed = args.seed

    # if results exist then skip
    if alp == 0.0 and os.path.exists(f'src/community_detection/extra_algs/dgcluster/results/results_{dataset_name}_{lam}_{epochs}_{base_model}_{seed}.json'):
        print(f'src/community_detection/extra_algs/dgcluster/results/results_{dataset_name}_{lam}_{epochs}_{base_model}_{seed}.pt exists. Skipping...')
        exit()
    elif alp != 0.0 and os.path.exists(f'src/community_detection/extra_algs/dgcluster/results/results_{dataset_name}_{lam}_{alp}_{epochs}_{base_model}_{seed}.json'):
        print(f'src/community_detection/extra_algs/dgcluster/results/results_{dataset_name}_{lam}_{alp}_{epochs}_{base_model}_{seed}.pt exists. Skipping...')
        exit()

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # device selection
    if torch.cuda.is_available() and device != 'cpu':
        device = torch.device(device)
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # transform data
    transform = T.NormalizeFeatures()

    # load dataset
    data = load_dataset(dataset_name)
    data = data.to(device)
    #print(data)

    # preprocessing
    num_nodes = data.num_nodes
    num_edges = (data.edge_index.shape[1])

    sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges), data.edge_index.cpu().numpy()), shape=(num_nodes, num_nodes))
    torch_sparse_adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(num_edges).to(device), size=(num_nodes, num_nodes))
    degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float().to(device)
    Graph = nx.from_scipy_sparse_array(sparse_adj, create_using=nx.Graph).to_undirected()
    num_edges = int((data.edge_index.shape[1]) / 2)

    in_dim = 128
    out_dim = 64
    model = GNN(num_nodes, in_dim, out_dim, base_model=base_model).to(device)

    optimizer_name = "Adam"
    lr = 1e-3
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001, amsgrad=True)

    train(model, optimizer, data, epochs, lam, alp)

    test_data = data.clone()
    print(test_data)

    model.eval()
    x = model(test_data)

    clusters = Birch(n_clusters=None, threshold=0.5).fit_predict(x.detach().cpu().numpy(), y=None)
    FQ = compute_fast_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree, device)
    print('No of clusters: ', max(clusters) + 1)
    print('Modularity:', FQ)

    results = {
        'num_clusters': np.unique(clusters).shape[0],
        'modularity': FQ,
    }

    if not os.path.exists('src/community_detection/extra_algs/dgcluster/results'):
        os.makedirs('src/community_detection/extra_algs/dgcluster/results')

    if not os.path.exists('src/community_detection/extra_algs/dgcluster/models'):
        os.makedirs('src/community_detection/extra_algs/dgcluster/models')
    
    if alp == 0.0:
        with open(f'src/community_detection/extra_algs/dgcluster/results/results_{dataset_name}_{lam}_{epochs}_{base_model}_{seed}.json', 'w') as f:
            json.dump(results, f)
        torch.save(model.state_dict(), f'src/community_detection/extra_algs/dgcluster/models/model_{dataset_name}_{lam}_{epochs}_{base_model}_{seed}.pth')
    else:
        with open(f'src/community_detection/extra_algs/dgcluster/results/results_{dataset_name}_{lam}_{alp}_{epochs}_{base_model}_{seed}.json', 'w') as f:
            json.dump(results, f)
        torch.save(model.state_dict(), f'src/community_detection/extra_algs/dgcluster/models/model_{dataset_name}_{lam}_{alp}_{epochs}_{base_model}_{seed}.pth')
