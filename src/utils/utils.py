from enum import Enum
from typing import List
import igraph as ig
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from statistics import mean
from hydra.core.hydra_config import HydraConfig


class FilePaths(Enum):
    """Class to store file paths for data and models"""

    # Local
    DATASETS_DIR = "dataset/networks"
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

    # Trained model path for testing (change the following line to change the model)
    TRAINED_MODEL = "src/methods/drl_agent/models/steps-10000_words-gre_eps-0_model.pth"

class DatasetFullNames(Enum):
    """Enum class for the dataset names"""

    KAR = "Zachary Karate Club"
    WORDS = "David Copperfield Words"
    VOTE = "Wikipedia Voting"
    POW = "U.S. Power Grid"
    FB_75 = "Facebook Friendships"
    COND_MAT = "Condense Matter Collaborations"
    FB_ART = "Facebook Artist Pages"
    DBLP = "DBLP Collaboration Network"
    YT = "Youtube Social Network"

class DatasetNames(Enum):
    """Enum class for the dataset names"""

    KAR = "kar"
    WORDS = "words"
    VOTE = "vote"
    POW = "pow"
    FB_75 = "fb-75"
    COND_MAT = "cond-mat"
    FB_ART = "fb-artist"
    DBLP = "dblp"
    YT = "youtube"
   
class DetectionAlgorithmsNames(Enum):
    """Enum class for the detection algorithms"""
    
    GRE = "greedy"
    LOUV = "louvain"
    WALK = "walktrap"
    LEID = "leiden"
    INF = "infomap"
    LAB = "label_propagation"
    EIG = "leading_eigenvector"
    BTW = "edge_betweenness"
    SPIN = "spinglass"

class EvasionAlgorithmsNames(Enum):
    RAND = "random"
    DEG = "degree"
    BETW = "betweenness"
    ROAM = "roam"
    DICE = "dice"
    NABLA = "nabla-cmh"
    DRL = "drl-agent"
    GRE = "greedy"

class ExperimentHyps(Enum):
    """Enum class for the experiment hyperparameters"""

    seed: int = 22
    target_community_size: List[int] = [0.2,0.5,0.8]
    max_steps_community_eval: int = 100

class DRL_agentHyps(Enum):
    LAMBDA = [0.1]
    ALPHA = [0.7]
    EPSILON = [0]
    EMBEDDING_DIM = 128  # 256
    WALK_NUMBER = 5  # 5, 10
    WALK_LENGTH = 40  # 40, 80
    HIDDEN_SIZE_1 = 64
    HIDDEN_SIZE_2 = 64
    DROPOUT = 0.2
    WEIGHT_DECAY = 1e-3
    EPS_CLIP = np.finfo(np.float32).eps.item()  # 0.2
    BEST_REWARD = -np.inf
    LR = [7e-4]
    GAMMA = [0.95]
    LR_EVAL = 0.0001  # LR[0]
    GAMMA_EVAL = 0.7  # GAMMA[0]
    LAMBDA_EVAL = 0.1  # LAMBDA[0]
    ALPHA_EVAL = 0.7  # ALPHA[0]
    EPSILON_EVAL = 25  # EPSILON[0]

class iGraphRNG:
    """
    Customized RNG to fix randomnees in iGraph
    """
    def __init__(self, seed: int = ExperimentHyps.seed.value):
        self.generator = random.Random(seed)
    
    def random(self):
        return self.generator.random()
    
    def randint(self, a:int , b: int):
        return self.generator.randint(a, b)
    
    def gauss(self, mu:float, sigma:float):
        return self.generator.gauss(mu, sigma)

class SimilarityFunctionsNames(Enum):
    """ Enum class for the similarity functions """

    # Community similarity function
    SOR = "sorensen"
    # Graph similarity function
    JAC = "jaccard"


class Utils:
    """Class to store utility functions"""

    def pre_process_graph(file_path:str) -> None:
        """
        Pre-process the graph by removing self-loops,multiple edges, and consider only the biggest connected component.
        The graph is saved in the same file.

        Parameters
        ----------
        file_path : str
            File path of the .txt file
        """
        if not file_path.endswith(".txt"):
            raise ValueError("File format not supported")
        
        # Load raw edge list
        with open(file_path, "r") as f:
            edges = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
        # Extract all unique nodes
        unique_nodes = sorted(set([node for edge in edges for node in edge]))
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
        reverse_node_mapping = {new_id: old_id for new_id, old_id in enumerate(unique_nodes)}
        # Relabel edges with new node ids
        relabeled_edges = [(node_mapping[src], node_mapping[dst]) for src, dst in edges]
        # Create graph from relabeled edge list
        graph = ig.Graph(edges=relabeled_edges, directed=False)
        graph = graph.simplify(multiple=True, loops=True)
        # Store original node IDs as 'name' attribute
        graph.vs["name"] = unique_nodes 
        # Return the biggest connected component
        largest_component = graph.clusters().giant()

        with open(file_path, "w") as f:
            for edge in largest_component.get_edgelist():
                f.write(f"{reverse_node_mapping[edge[0]]} {reverse_node_mapping[edge[1]]}\n")
    
    @staticmethod
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

    @staticmethod
    def check_dir(path: str) -> None:
        """
        Check if the directory exists, if not create it.

        Parameters
        ----------
        path : str
            Path to the directory
        """
        if not os.path.exists(path):
            os.makedirs(path)
    
    @staticmethod
    def plot_metrics(
        datasets: List[str],
        evasion_algs: List[str],
        detection_algs: List[str],
        budget_factors: List[float],
        taus: List[float],
        metrics: List[str],
    ) -> None:
        """
        Plot the metrics for the evasion algorithms.

        Parameters
        ----------
        datasets : List[str]
            List of datasets
        evasion_algs : List[str]
            List of evasion algorithms
        detection_algs : List[str]
            List of detection algorithms
        budget_factors : List[float]
            List of budget factors
        taus : List[float]
            List of tau values
        metrics : List[str]
            List of metrics
        output_dir : str
            Output directory
        """

        output_dir = HydraConfig.get().runtime.output_dir + "/"
        plots_dir = "/plots/"
        for dataset in datasets:
            dataset_name = getattr(DatasetNames, dataset).value
            for alg in detection_algs:
                alg_name = getattr(DetectionAlgorithmsNames, alg).value
                for tau in taus:
                    for c_beta in budget_factors:

                        # ----------------- Load the results ----------------- #
                        results_dir = output_dir + f"{dataset_name}/{alg_name}/tau_{tau}/betaFactor_{c_beta}/json_results/"
                        output_plots_dir = output_dir + f"{dataset_name}/{alg_name}/tau_{tau}/betaFactor_{c_beta}" + plots_dir
                        Utils.check_dir(output_plots_dir)
                        results = {}
                        for evasion_alg in evasion_algs:
                            evasion_alg_name = getattr(EvasionAlgorithmsNames, evasion_alg).value
                            file_name = results_dir + f"{evasion_alg_name}.json"
                            with open(file_name, "r", encoding="utf-8") as f:
                                log = json.load(f)
                            results[evasion_alg] = log
                        budget = max(results[evasion_algs[0]]["steps"]) # for steps plot

                        # ----------------- Store/compute metrics ----------------- #
                        metrics_data = {}
                        for metric in metrics:
                            if metric == "f1":
                                df = pd.DataFrame(
                                    {
                                        "Algorithm": evasion_algs,
                                        metric.capitalize(): [
                                            0 if (mean(results[alg]["goal"]) + mean(results[alg]["nmi"])) == 0 else 
                                            2 * mean(results[alg]["goal"]) * mean(results[alg]["nmi"]) / 
                                            (mean(results[alg]["goal"]) + mean(results[alg]["nmi"]))
                                            for alg in evasion_algs
                                        ],
                                    }
                                )
                            elif metric == "steps":
                                df = pd.DataFrame(
                                    {
                                        "Algorithm": evasion_algs,
                                        metric.capitalize(): [
                                            mean([results[alg][metric][i] for i in range(len(results[alg]["goal"])) if results[alg]["goal"][i] == 1])/budget 
                                            if any(results[alg]["goal"][i] == 1 for i in range(len(results[alg]["goal"]))) else 0 
                                            for alg in evasion_algs],
                                    }
                                )
                            else:
                                df = pd.DataFrame(
                                    {
                                        "Algorithm": evasion_algs,
                                        metric.capitalize(): [mean(results[alg][metric]) for alg in evasion_algs],
                                    }
                                )
                            # Convert the goal column to percentage
                            if metric == "goal":
                                df[metric.capitalize()] = df[metric.capitalize()] * 100
                            # Convert the budget column to percentage
                            if metric == "steps":
                                df[metric.capitalize()] = df[metric.capitalize()] * 100
                            
                            # Store metric data for JSON
                            metrics_data[metric] = df.set_index("Algorithm").to_dict()[metric.capitalize()]

                            # ----------------- Plot ----------------- #
                            if len(evasion_algs) > 1:
                                sns.barplot(
                                    data=df,
                                    x="Algorithm",
                                    y=metric.capitalize(),
                                    hue="Algorithm",
                                    palette=sns.color_palette("tab10"),
                                    edgecolor="black",  
                                    linewidth=0.5
                                )
                                plt.title(
                                    f"Evaluation on {dataset_name} graph with {alg_name} algorithm"
                                )
                                plt.xlabel("Algorithm")
                                if metric == "goal":
                                    plt.ylabel(f"{metric.capitalize()} reached %")
                                elif metric == "time":
                                    plt.ylabel(f"{metric.capitalize()} (s)")
                                elif metric == "steps":
                                    plt.ylabel("Budget used % if goal reached")
                                else:
                                    plt.ylabel(metric.capitalize())
                                file_path = output_plots_dir + f"{metric}.png"
                                plt.savefig(file_path)
                                plt.clf()

                        # Save metrics data to JSON
                        metrics_json_path = output_plots_dir + "metrics.json"
                        with open(metrics_json_path, "w", encoding="utf-8") as json_file:
                            json.dump(metrics_data, json_file, indent=4)

                    















