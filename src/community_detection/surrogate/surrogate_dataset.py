import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import igraph as ig
from src.community_detection.algorithms import CommunityDetectionAlg
from src.baselines.degree import DegreeHiding
from src.community_detection.extra_algs.dgcluster.dgcluster import DGCluster
from src.community_detection.extra_algs.locale.locale import ig_leiden_locale
from cdlib import NodeClustering
from sklearn.cluster import AgglomerativeClustering
from torch_geometric.data import Data
import numpy as np
import random

def ensemble_clustering(clusterings, num_nodes):
    """
    clusterings: list of dictionaries node_id -> cluster_id (one for each algorithm)
    num_nodes: total number of nodes
    """
    # Step 1: build the coassociation matrix
    coassoc = np.zeros((num_nodes, num_nodes))
    for clustering in clusterings:
        labels = [clustering.get(i, -1) for i in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if labels[i] != -1 and labels[i] == labels[j]:
                    coassoc[i][j] += 1

    # Normalize by the number of algorithms
    coassoc /= len(clusterings)

    # Step 2: hierarchical clustering on the dissimilarity matrix
    dissimilarity = 1.0 - coassoc
    avg_n_clusters = int(np.mean([len(c) for c in clusterings]))
    model = AgglomerativeClustering(
        n_clusters=avg_n_clusters,
        metric='precomputed',
        linkage='average'
    )
    final_labels = model.fit_predict(dissimilarity)

    return final_labels

def nodeclustering_to_dict(nodeclustering, n_nodes):
    labels = np.zeros(n_nodes, dtype=int) - 1
    for idx, comm in enumerate(nodeclustering.communities):
        for node in comm:
            labels[node] = idx
    return {i: labels[i] for i in range(n_nodes)}

def generate_surrogate_dataset(graph: ig.Graph, env, max_nodes=500, max_modifications=3, seed=22):
    """
    Generates the surrogate dataset as described.
    """
    random.seed(seed)
    np.random.seed(seed)
    n_nodes = graph.vcount()
    nodes = list(range(n_nodes))
    if n_nodes > max_nodes:
        nodes = random.sample(nodes, max_nodes)
    # Original clusters
    leiden = CommunityDetectionAlg('leiden', env).community_detection(graph)
    walktrap = CommunityDetectionAlg('walktrap', env).community_detection(graph)
    dgcluster = DGCluster(graph, env.graph_name)
    clusterings = [
        nodeclustering_to_dict(leiden, n_nodes),
        nodeclustering_to_dict(walktrap, n_nodes),
        nodeclustering_to_dict(dgcluster, n_nodes)
    ]
    ensemble = ensemble_clustering(clusterings, n_nodes)
    dataset = []
    # Append the original graph
    dataset.append({
        'target_node': None,
        'num_modifications': 0,
        'perturbed_graph': graph,
        'agg_clusters': ensemble
    })
    for target_node in nodes:
        for num_mod in range(1, max_modifications+1):
            dh = DegreeHiding(env, target_node, num_mod)
            pert_graph, _ , _ = dh.community_membership_hiding()
            # Recalculate the clusters
            leiden_p = CommunityDetectionAlg('leiden', env).community_detection(pert_graph)
            walktrap_p = CommunityDetectionAlg('walktrap', env).community_detection(pert_graph)
            dgcluster_p = DGCluster(pert_graph, env.graph_name)
            clusterings_p = [
                nodeclustering_to_dict(leiden_p, n_nodes),
                nodeclustering_to_dict(walktrap_p, n_nodes),
                nodeclustering_to_dict(dgcluster_p, n_nodes)
            ]
            ensemble_p = ensemble_clustering(clusterings_p, n_nodes)
            dataset.append({
                'target_node': target_node,
                'num_modifications': num_mod,
                'perturbed_graph': pert_graph,
                'agg_clusters': ensemble_p
            })
    return dataset 