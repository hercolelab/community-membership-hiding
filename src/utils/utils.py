from enum import Enum
from typing import List, Tuple
from statistics import mean, stdev
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import pandas as pd
import random
import seaborn as sns
import scipy.io
import json
import os
import math


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

    # Trained model path for testing (change the following line to change the model)
    #TRAINED_MODEL = "src/models/steps-10000_words-gre_eps-0_model.pth"

class DatasetFullNames(Enum):
    """Enum class for the dataset names"""

    KAR = "Zachary Karate Club"
    WORDS = "David Copperfield Words"
    VOTE = "Wikipedia Voting"
    POW = "U.S. Power Grid"
    FB_75 = "Facebook Friendships"
    COND_MAT = "Condense Matter Collaborations"

class DatasetNames(Enum):
    """Enum class for the dataset names"""

    KAR = "kar"
    WORDS = "words"
    VOTE = "vote"
    POW = "pow"
    FB_75 = "fb-75"
    COND_MAT = "cond-mat"
   
class DetectionAlgorithmsNames(Enum):
    """Enum class for the detection algorithms"""

    GRE = "greedy"
    INF = "infomap"
    LAB = "label_propagation"
    LOUV = "louvain"
    WALK = "walktrap"

class ExperimentHyps(Enum):
    """Enum class for the experiment hyperparameters"""

    seed: int = 22
    target_community_size: List[int] = [0.2,0.5,0.8]
    max_steps_community_eval: int = 100

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



class Utils:
    """Class to store utility functions"""
    
    @staticmethod
    def import_graph(file_path: str) -> ig.Graph:
        """
        Import an unweighted graph from a txt file using igraph

        Parameters
        ----------
        file_path : str
            File path of the .txt file
            
        Returns
        -------
        ig.Graph
            Graph imported from the file path
        """
        if file_path.endswith(".txt"):
            graph = ig.Graph.Read_Edgelist(file_path, directed=False)
            graph = graph.simplify(multiple=True, loops=True)
        else:
            raise ValueError("File format not supported")
        
        # Remove nodes with degree 0
        ## Igraph starts from 0 index, so if txt file starts from 1, we need to delete the first vertex
        graph.delete_vertices(graph.vs.select(_degree_eq=0))
        return graph

    @staticmethod
    def check_dir(path: str):
        """
        Check if the directory exists, if not create it.

        Parameters
        ----------
        path : str
            Path to the directory
        """
        if not os.path.exists(path):
            os.makedirs(path)
    

# ------ Example usage for the Enums ------ #
#dataset = "KAR"
#file_path = getattr(FilePaths, dataset).value
#dataset_name = getattr(DatasetNames, dataset).value
#print(f"File path for {dataset_name}: {file_path}")