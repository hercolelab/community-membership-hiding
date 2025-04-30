from src.graph_environment.env import GraphEnvironment
from src.utils.utils import EvasionAlgorithmsNames, DRL_agentHyps, FilePaths, ExperimentHyps
from src.community_detection.algorithms import CommunityDetectionAlg
import logging
import hydra
import yaml
from src.baselines.random import RandomHiding
from src.baselines.degree import DegreeHiding
from src.baselines.betweenness import CentralityHiding
from src.baselines.roam import RoamHiding
from src.baselines.dice import DiceHiding
from src.methods.nabla_cmh.nabla_cmh import nablaCMH
from src.methods.drl_agent.agent import Agent
from cdlib import NodeClustering
from collections import defaultdict
from time import time
import json
from typing import Tuple, List
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import random
from karateclub import Node2Vec
import numpy as np
from rich.progress import Progress
import matplotlib.pyplot as plt


"""
# ------ EVASION OPTIONS ------ #
This script analyze robustness of Node2Vec embeddings and associated DGCluster performances.

Available Datasets:
    - KAR      
    - WORDS     
    - VOTE      
    - POW       
    - FB_75     
    - COND_MAT

"""

# ------ UTILS FUNCTIONS ------ #
def get_memberships(clusters: NodeClustering) -> List[int]:
    """
    Get the memberships of the nodes in the clusters

    Parameters
    ----------
    clusters : NodeClustering
        The clusters to get the memberships from

    Returns
    -------
    memberships : list
        The memberships of the nodes in the clusters
    """
    memberships = []
    temp_memberships = defaultdict(list, sorted(clusters.to_node_community_map().items()))
    for node, comm in temp_memberships.items():
        memberships.append(comm[0])
    return memberships

def get_distances(embedding1: np.array, embedding2: np.array) -> Tuple[float,float]:
    """
    Get the distances between the embeddings

    Parameters
    ----------
    embedding1 : np.array
        The first embedding
    embedding2 : np.array
        The second embedding

    Returns
    -------
    mean_distance : float
        The mean distance between the embeddings
    median_distance : float
        The median distance between the embeddings
    """
    node_distances = np.linalg.norm(embedding1 - embedding2, axis=1)
    mean_distance = np.mean(node_distances)
    median_distance = np.median(node_distances)
    return mean_distance, median_distance

def convert_numpy_types(obj):
    """
    Converte i tipi numpy in tipi Python standard per la serializzazione JSON
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj



# ------ ANALYSIS CONFIGURATION ------ #
graph_name = "WORDS"

# ------ UPDATE HYDRA CONFIG FILE ------ #
with open("src/conf/robustness.yaml", "r") as file:
        cfg = yaml.safe_load(file)
cfg["graph"] = graph_name
with open("src/conf/robustness.yaml", "w") as file:
    yaml.dump(cfg, file, sort_keys=False)


# ------ MAIN FUNCTION ------ #
log = logging.getLogger(__name__)
@hydra.main(config_path="src/conf", config_name="robustness", version_base=None)
def main(cfg: DictConfig) -> None:

    # Set the environment, graph, and detection algorithm
    env = GraphEnvironment(graph_name, ["GRE"])
    original_graph = env.original_graph.copy()
    original_nx_graph = original_graph.to_networkx()
    preferred_sizes = ExperimentHyps.target_community_size.value
    detection_alg = CommunityDetectionAlg("dgcluster", env)

    # Set the embedding model
    embedding_model = Node2Vec()
    # Compute the first embeddings and community structure
    original_level = logging.root.level
    logging.root.setLevel(logging.ERROR)
    embedding_model.fit(original_nx_graph.copy())
    original_embedding = embedding_model.get_embedding()
    logging.root.setLevel(original_level)
    original_communities = detection_alg.community_detection(original_graph)

    # Results dictionary
    results = {
        "graph": env.graph_name_output,
        "target_node": [],
        "mean_embeddings_distance": [],
        "median_embeddings_distance": [],
        "number_of_communities": [],
        "nmi_score": [],
        "modularity": [],
        "steps": [],
        "changes": [],
    }
    
    # Print info
    log.info("========== ANALYZING ROBUSTNESS OF Node2Vec AND DGCluster  ==========") 
    log.info("  • Graph   : %s", env.graph_name_output)
    log.info(">> Starting the analysis...")

    with Progress() as progress:
        outer_task = progress.add_task("[cyan]Community Steps...", total=len(preferred_sizes))
        inner_task = None
        inner_inner_task = None
            
        for i in range(len(preferred_sizes)):

            progress.update(
                outer_task,
                description=f"[cyan] Community Step {i+1}/3"
            )
            
            # Set the preferred community size to the environment
            env.preferred_community_size = preferred_sizes[i]
            # Change the target community
            env.change_target_community()
            experiment_steps = len(env.list_target_nodes)

            inner_task = progress.add_task(
                f"[green]Processing Nodes for Community {i+1}...",
                total=experiment_steps
            )   

            for j in range(experiment_steps):

                progress.update(
                    inner_task,
                    description=f"[green] Node Step {j+1}/{experiment_steps}"
                )

                # Change target node within the target community
                env.change_target_node()
                target_node = env.target_node
                results["target_node"].append(target_node)

                mean_dist = []
                median_dist = []
                comm = []
                nmi = []
                mod = []
                s = []
                c = []

                max_budget = 10

                inner_inner_task = progress.add_task(
                    f"[magenta]Processing Node {target_node}...",
                    total=max_budget+1
                )   
                
                for budget in range(0, max_budget+1):

                    progress.update(
                        inner_inner_task,
                        description=f"[magenta] Budget {budget}"
                    )

                    # Set the budget to the environment
                    env.budget = budget
                    # Run a evasion algorithm. For instance, Degree baseline
                    evasion_alg = DegreeHiding(env, target_node, env.budget)
                    new_graph, steps, changes = evasion_alg.community_membership_hiding()

                    # Compute the new embeddings
                    logging.root.setLevel(logging.ERROR)
                    embedding_model.fit(new_graph.to_networkx().copy())
                    new_embedding = embedding_model.get_embedding()
                    logging.root.setLevel(original_level)
                    # Compute the distances between the embeddings
                    mean_distance, median_distance = get_distances(original_embedding, new_embedding)
                    # Compute the new community structure
                    new_communities = detection_alg.community_detection(new_graph)
                    # Compute the modularity
                    new_memberships = get_memberships(new_communities)
                    modularity = new_graph.modularity(new_memberships)
                    # Compute the number of communities
                    number_of_communities = len(new_communities.communities)
                    # Compute the NMI score
                    nmi_score = original_communities.normalized_mutual_information(new_communities).score

                    # Save the results
                    mean_dist.append(mean_distance)
                    median_dist.append(median_distance)
                    comm.append(number_of_communities)
                    nmi.append(nmi_score)
                    mod.append(modularity)
                    s.append(steps)
                    c.append(changes)

                    progress.update(inner_inner_task, advance=1)

                # Save the results to the dictionary
                results["mean_embeddings_distance"].append(mean_dist)
                results["median_embeddings_distance"].append(median_dist)
                results["number_of_communities"].append(comm)
                results["nmi_score"].append(nmi)
                results["modularity"].append(mod)
                results["steps"].append(s)
                results["changes"].append(c)

                progress.remove_task(inner_inner_task)
                progress.update(inner_task, advance=1)
            progress.update(
                outer_task,
                description=f"[cyan] Community Step {i+1}/3"
            )
            progress.update(outer_task, advance=1)
    
    # Save the results to a JSON file
    output_dir = HydraConfig.get().runtime.output_dir + "/"
    output_file = f"{output_dir}robustness_results_{graph_name}.json"
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4, default=convert_numpy_types)

    log.info(">> Results saved to %s", output_file)
    log.info(">> Computing metrics and plots...")

    # Compute the plots
    avg_mean_embeddings_distance = [
        np.mean([mean_dist[i] for mean_dist in results["mean_embeddings_distance"]])
        for i in range(len(results["mean_embeddings_distance"][0]))
    ]
    avg_median_embeddings_distance = [
        np.mean([mean_dist[i] for mean_dist in results["median_embeddings_distance"]])
        for i in range(len(results["mean_embeddings_distance"][0]))
    ]
    avg_number_of_communities = [
        np.mean([comm[i] for comm in results["number_of_communities"]])
        for i in range(len(results["number_of_communities"][0]))
    ]
    avg_nmi_score = [
        np.mean([nmi[i] for nmi in results["nmi_score"]])
        for i in range(len(results["nmi_score"][0]))
    ]
    avg_modularity = [
        np.mean([mod[i] for mod in results["modularity"]])
        for i in range(len(results["modularity"][0]))
    ]

    # Create x-axis values
    x_values = range(0, max_budget + 1)
    # Plot average mean embeddings distance
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, avg_mean_embeddings_distance, marker='o', label='Mean Embeddings Distance')
    plt.xlabel('Budget')
    plt.ylabel('Mean Embeddings Distance')
    plt.title(f'Mean Embeddings Distance of {graph_name}')
    plt.grid()
    plt.savefig(f"{output_dir}mean_embeddings_distance_{graph_name}.png")

    # Plot average median embeddings distance
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, avg_median_embeddings_distance, marker='o', label='Median Embeddings Distance')
    plt.xlabel('Budget')
    plt.ylabel('Median Embeddings Distance')
    plt.title(f'Median Embeddings Distance of {graph_name}')
    plt.grid()
    plt.savefig(f"{output_dir}median_embeddings_distance_{graph_name}.png")

    # Plot average number of communities
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, avg_number_of_communities, marker='o', label='Number of Communities')
    plt.xlabel('Budget')
    plt.ylabel('Number of Communities')
    plt.title(f'Number of Communities of {graph_name}')
    plt.grid()
    plt.savefig(f"{output_dir}number_of_communities_{graph_name}.png")

    # Plot average NMI score
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, avg_nmi_score, marker='o', label='NMI Score')
    plt.xlabel('Budget')
    plt.ylabel('NMI Score')
    plt.title(f'NMI Score of {graph_name}')
    plt.grid()
    plt.savefig(f"{output_dir}nmi_score_{graph_name}.png")

    # Plot average modularity
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, avg_modularity, marker='o', label='Modularity')
    plt.xlabel('Budget')
    plt.ylabel('Modularity')
    plt.title(f'Modularity of {graph_name}')
    plt.grid()
    plt.savefig(f"{output_dir}modularity_{graph_name}.png")

    log.info(">> Analysis completed.")
    log.info("========== END OF ANALYSIS ==========")


if __name__ == "__main__":
    main()
