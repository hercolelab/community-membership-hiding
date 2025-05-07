import igraph as ig 
from typing import Tuple, List
import numpy as np
import pandas as pd
import time as time
from hydra.core.hydra_config import HydraConfig
import logging
from src.utils.utils import Utils, iGraphRNG, DetectionAlgorithmsNames, DatasetNames, DatasetFullNames
import os
import scipy as sp
import json
from collections import defaultdict
import hydra
import yaml
import torch
from omegaconf import DictConfig
import leidenalg as la
from src.community_detection.extra_algs.scd import ig_SCD
from src.community_detection.extra_algs.locale.locale import ig_leiden_locale
from src.community_detection.extra_algs.dgcluster.dgcluster import DGCluster
from dgcluster_training import compute_fast_modularity, from_ig_graph_to_node2vec_geometric_data

""" 
Available Datasets (only one can be selected):
    - KAR: Zachary Karate Club    
    - WORDS: David Copperfield Words    
    - VOTE: Wikipedia Voting    
    - POW: U.S. Power Grid   
    - FB_75: Facebook Friendships
    - COND_MAT: Condense Matter Collaborations
    - FB_ART : Facebook Artist Pages
    - DBLP : DBLP Collaborations
    - YT: YouTube Social Network

Available Community Detection Algorithms (multiple can be selected):
    - GRE:  Greedy
    - LOUV: Louvain
    - WALK: Walktrap
    - LEID: Leiden
    - INF:  Infomap
    - LAB:  Label Propagation
    - EIG:  Leading Eigenvector
    - BTW:  Edge Betweenness
    - SPIN: Spinglass
    - SCD: Scalable Community Detection
    - LOC: Locale
    - DGC: DGCluster
    
"""

# ------ EXPERIMENT CONFIGURATION ------ #
graph = "KAR"

#detection_algs = ["GRE", "LOUV", "LEID", "WALK", "INF", "LAB", "EIG", "BTW", "SPIN", "SCD", "LOC"]
# For large graphs, do not use "BTW" since in n^3 complexity
detection_algs = ["GRE", "LOUV", "LEID", "WALK", "INF", "SPIN", "SCD", "LOC", "DGC"]



# ------ UPDATE HYDRA CONFIG FILE ------ #
root_dir = os.path.abspath(os.path.join('./'))
outputs_dir = root_dir + "/outputs_review/dataset_analysis/"
with open(f"{root_dir}/src/conf/dataset_analysis.yaml", "r") as file:
        cfg = yaml.safe_load(file)
cfg["graph"] = graph
cfg["community_detection_algs"] = detection_algs
with open(f"{root_dir}/src/conf/dataset_analysis.yaml", "w") as file:
    yaml.dump(cfg, file, sort_keys=False)


log = logging.getLogger(__name__)
@hydra.main(config_path="src/conf", config_name="dataset_analysis", version_base=None)
def main(cfg:DictConfig) -> None:
    """
    Community detection analysis on the graph

    Parameters
    ----------
    graph : str
        Name of the graph to analyze
    
    detection_algs : List[str]
        List of community detection algorithms to apply
    """

    # Import graph
    graph_name = getattr(DatasetNames, graph).value
    graph_full_name = getattr(DatasetFullNames, graph).value
    graph_path = root_dir + "/dataset/networks/" + graph_name + ".txt"

    log.info(f"Analysis for network {graph_name} started.")

    start_time = time.time()
    G = Utils.import_graph(graph_path)
    end_time = time.time()
    import_time = round(end_time - start_time, 6)

    # Set up results storage
    log_name = graph_name + ".json"
    log_path = outputs_dir + log_name
    log_exists = os.path.exists(log_path)
    if log_exists:
        with open(log_path, 'r') as file:
            results = json.load(file)
    else:
        results = {}
    
    # Graph Analysis
    if not log_exists:
        n_nodes = G.vcount()
        n_edges = G.ecount()
        degrees = G.degree()
        diameter = G.diameter()
        avg_degree = round(sum(degrees) / len(degrees), 3)
        max_degree = max(degrees)
        avg_path_length = round(G.average_path_length(), 3)

        results = {
            "name": graph_full_name,
            "import_time": import_time,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "diameter": diameter,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "avg_path_length": avg_path_length,
            "detection_algorithms": {}
        }


    # Community Detection algorithms
    algorithms = {
        "GRE": G.community_fastgreedy,
        "LOUV": G.community_multilevel,
        "WALK": G.community_walktrap,
        "LEID": la.find_partition,
        "INF": G.community_infomap,
        "LAB": G.community_label_propagation,
        "EIG": G.community_leading_eigenvector,
        "BTW": G.community_edge_betweenness,
        "SPIN": G.community_spinglass,
        "SCD": ig_SCD(iterations=30),
        "LOC": ig_leiden_locale,
        "DGC": DGCluster,
    }

    # Set the random number generator for reproducibility
    custom_rng = iGraphRNG()
    ig.set_random_number_generator(custom_rng)

    for alg in detection_algs:
        
        alg_name = getattr(DetectionAlgorithmsNames, alg).value
        algorithm = algorithms[alg]
        
        start_time = time.time()
        
        try:

            avg_modularity = 0
            avg_communities = 0
            avg_time = 0

            for i in range(10):

                if alg in ["GRE", "WALK", "BTW"]:
                    communities = algorithm().as_clustering()
                elif alg == "LEID":
                    communities = algorithm(G, la.ModularityVertexPartition)
                elif alg == "SCD":
                    algorithm.fit(G)
                    communities = algorithm.get_memberships()
                elif alg == "LOC":
                    communities = algorithm(G, "KAR")
                elif alg == "DGC":
                    communities = algorithm(G, graph)
                else:
                    communities = algorithm()

                end_time = time.time()
                detect_time = round(end_time - start_time, 6)
                avg_time += detect_time

                if alg in ["SCD", "LOC", "DGC"]:
                    memberships = []
                    temp_memberships = defaultdict(list, sorted(communities.to_node_community_map().items()))
                    for node, comm in temp_memberships.items():
                        memberships.append(comm[0])
                else:
                    memberships = communities.membership

                # Compute modularity
                modularity = G.modularity(memberships)
                avg_modularity += modularity
                # Compute number of communities
                if alg not in ["SCD", "LOC", "DGC"]:
                    avg_communities += len(communities)
                else:
                    avg_communities += len(communities.communities)

            log.info(f"- Algorithm {alg_name} executed correctly.")

            results["detection_algorithms"][alg_name] = {
                "Number of Communities": int(avg_communities / 10),
                "Modularity": round(avg_modularity / 10, 6),
                "Time": round(avg_time / 10, 6)
            }

        except ig.InternalError as e:
            log.info(f"Algorithm {alg_name} failed: {e}")
            results["detection_algorithms"][alg_name] = {
                "Number of Communities": "NA",
                "Time": "NA"
            }


    # Save results to JSON file
    save_path = HydraConfig.get().runtime.output_dir + "/" + log_name
    with open(save_path, 'w') as file:
        json.dump(results, file, indent=4)
    log.info(f"Results saved to {save_path}")
    log.info(f"Analysis for network '{graph_name}' completed.")


if __name__ == "__main__":
    main()