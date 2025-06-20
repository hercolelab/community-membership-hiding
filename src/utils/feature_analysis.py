import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import igraph as ig
import numpy as np
import pandas as pd
import time as time
from hydra.core.hydra_config import HydraConfig
import logging
from src.utils.utils import Utils, DatasetNames, DatasetFullNames
import yaml
from omegaconf import DictConfig
import hydra
import seaborn as sns
import matplotlib.pyplot as plt

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

Available Node Features (multiple can be selected):
    - degree
    - eccentricity
    - betweenness_centrality
    - closeness_centrality
    - eigenvector_centrality
    - coreness_centrality
    - harmonic_centrality
    - clustering_coefficient
    - pagerank
"""

# ------ EXPERIMENT CONFIGURATION ------ #
graph = "COND_MAT"
features = ["degree", "eccentricity", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality", "coreness_centrality", "harmonic_centrality", "clustering_coefficient", "pagerank"]

# ------ UPDATE HYDRA CONFIG FILE ------ #
root_dir = os.path.abspath(os.path.join('./'))
outputs_dir = root_dir + "/outputs_review/feature_analysis/"
with open(f"{root_dir}/src/conf/feature_analysis.yaml", "r") as file:
    cfg = yaml.safe_load(file)
cfg["graph"] = graph
cfg["features"] = features
with open(f"{root_dir}/src/conf/feature_analysis.yaml", "w") as file:
    yaml.dump(cfg, file, sort_keys=False)

log = logging.getLogger(__name__)
@hydra.main(config_path="../../src/conf", config_name="feature_analysis", version_base=None)
def main(cfg:DictConfig) -> None:
    """
    Node feature analysis on the graph

    Parameters
    ----------
    graph : str
        Name of the graph to analyze
    features : List[str]
        List of node features to compute
    """
    # Import graph
    graph_name = getattr(DatasetNames, graph).value
    graph_full_name = getattr(DatasetFullNames, graph).value
    graph_path = root_dir + "/dataset/networks/" + graph_name + ".txt"

    log.info(f"Feature analysis for network {graph_name} started.")

    G = Utils.import_graph(graph_path)

    # Set up results storage
    log_name = graph_name + "_features.csv"
    log_path = outputs_dir + "/" + graph_name + "/" + log_name
    if not os.path.exists(outputs_dir+"/"+graph_name):
        os.makedirs(outputs_dir+"/"+graph_name)

    # Node feature extraction
    node_ids = list(range(G.vcount()))
    data = {"node": node_ids}
    timings = {}

    if "degree" in features:
        start = time.time()
        data["degree"] = G.degree()
        timings["degree"] = time.time() - start
    if "eccentricity" in features:
        start = time.time()
        data["eccentricity"] = G.eccentricity()
        timings["eccentricity"] = time.time() - start
    if "betweenness_centrality" in features:
        start = time.time()
        data["betweenness_centrality"] = G.betweenness()
        timings["betweenness_centrality"] = time.time() - start
    if "closeness_centrality" in features:
        start = time.time()
        data["closeness_centrality"] = G.closeness()
        timings["closeness_centrality"] = time.time() - start
    if "eigenvector_centrality" in features:
        start = time.time()
        data["eigenvector_centrality"] = G.eigenvector_centrality()
        timings["eigenvector_centrality"] = time.time() - start
    if "coreness_centrality" in features:
        start = time.time()
        data["coreness_centrality"] = G.coreness()
        timings["coreness_centrality"] = time.time() - start
    if "harmonic_centrality" in features:
        start = time.time()
        data["harmonic_centrality"] = G.harmonic_centrality()
        timings["harmonic_centrality"] = time.time() - start
    if "clustering_coefficient" in features:
        start = time.time()
        data["clustering_coefficient"] = G.transitivity_local_undirected(mode="zero")
        timings["clustering_coefficient"] = time.time() - start
    if "pagerank" in features:
        start = time.time()
        data["pagerank"] = G.pagerank()
        timings["pagerank"] = time.time() - start
    # Add more features as needed

    df = pd.DataFrame(data)
    df.to_csv(log_path, index=False)
    log.info(f"Node features saved to {log_path}")

    # Save timings in a separate DataFrame
    timing_items = [(feat, timings.get(feat, None)) for feat in features]
    df_time = pd.DataFrame(timing_items, columns=["feature", "time_seconds"])
    log_time_path = outputs_dir + "/" + graph_name + "/" + graph_name + "_features_time.csv"
    df_time.to_csv(log_time_path, index=False)
    log.info(f"Feature computation times saved to {log_time_path}")

    # --- Correlation analysis and heatmaps ---
    # Remove 'node' column for correlation
    feature_df = df.drop(columns=["node"])

    # Rename columns for better heatmap labels
    pretty_names = {}
    for col in feature_df.columns:
        if col == "clustering_coefficient":
            pretty_names[col] = "loc_transitivity"
        elif col.endswith("_centrality"):
            pretty_names[col] = col.replace("_centrality", "")
        else:
            pretty_names[col] = col
    feature_df_renamed = feature_df.rename(columns=pretty_names)

    # Pearson correlation
    pearson_corr = feature_df_renamed.corr(method="pearson")
    pearson_corr_path = outputs_dir + f"/{graph_name}/{graph_name}_pearson_corr.csv"
    pearson_corr.to_csv(pearson_corr_path)
    # Spearman correlation
    spearman_corr = feature_df_renamed.corr(method="spearman")
    spearman_corr_path = outputs_dir + f"/{graph_name}/{graph_name}_spearman_corr.csv"
    spearman_corr.to_csv(spearman_corr_path)

    # Plot and save heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation Heatmap")
    pearson_heatmap_path = outputs_dir + f"/{graph_name}/{graph_name}_pearson_heatmap.png"
    plt.tight_layout()
    plt.savefig(pearson_heatmap_path)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Spearman Correlation Heatmap")
    spearman_heatmap_path = outputs_dir + f"/{graph_name}/{graph_name}_spearman_heatmap.png"
    plt.tight_layout()
    plt.savefig(spearman_heatmap_path)
    plt.close()

    log.info(f"Pearson and Spearman correlation matrices and heatmaps saved to {outputs_dir}/{graph_name}/")

    # --- Place for further analysis ---
    # TODO: Add further analysis on the node features DataFrame

if __name__ == "__main__":
    main() 