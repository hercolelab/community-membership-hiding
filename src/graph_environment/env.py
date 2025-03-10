from src.community_detection.algorithms import CommunityDetectionAlg
from src.utils.utils import Utils, FilePaths, DatasetNames, DatasetFullNames, DetectionAlgorithmsNames, ExperimentHyps
from typing import List, Tuple, Callable
import igraph as ig
import numpy as np
import cdlib
import random
import time
import torch
import copy

class GraphEnvironment(object):
    """Class for the Graph Environment"""

    def __init__(
            self, 
            graph_name: str,
            community_detection_alg: str,
        ) -> None:
        """
        Initialize the Graph Environment object

        Parameters
        ----------
        graph_name : str
            The name of the graph, e.g. "KAR"
        community_detection_alg : str
            The name of the community detection algorithm   
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = ExperimentHyps.seed

        # ------ GRAPH ------ #
        # Graph name
        self.graph_name: str = graph_name
        self.graph_name_output: str = getattr(DatasetNames, self.graph_name).value
        self.graph_name_full: str = getattr(DatasetFullNames, self.graph_name).value
        # Graph objects
        self.graph_agent: ig.Graph = None # Graph object for the agent, characterized by random features for the nodes embeddings
        self.original_graph: ig.Graph = None
        self.old_graph: ig.Graph = None # Graph before the last action, used by some methods to compute distances between graphs
        # Set the graph
        self.set_graph()

        # ------ COMMUNITY DETECTION ------ #
        # Community detection algorithm
        self.community_detection_alg_name: str = community_detection_alg
        self.community_detection_alg_name_output: str = getattr(DetectionAlgorithmsNames, self.community_detection_alg_name).value
        self.community_detection_alg: CommunityDetectionAlg = None
        # Communities
        self.original_communities: List[List[int]] = None
        self.old_communities: List[List[int]] = None # Communities before the last action, used by some methods to compute distances between communities
        self.new_communities: List[List[int]] = None
        # Set the community detection algorithm
        self.set_communities()

    
    # ============================================================================= #
    #                              SETTERS FUNCTIONS                                #
    # ============================================================================= #


    def set_graph(self) -> None:
        """Set the graph """
        
        self.original_graph = Utils.import_graph(getattr(FilePaths, self.graph_name).value)
        #self.graph_agent = self.set_graph_agent()

    def set_communities(self) -> None:
        """Set the community detection algorithm"""

        self.community_detection_alg = CommunityDetectionAlg(self.community_detection_alg_name_output)
        self.original_communities = self.community_detection_alg.community_detection(self.original_graph)
        self.old_communities = self.original_communities
        

    # ============================================================================= #
    #                              GETTERS FUNCTIONS                                #
    # ============================================================================= #

    ## TO DO


    # ============================================================================= #
    #                             ENVIRONMENT INFO                                  #
    # ============================================================================= #

    def print_environment_info(self) -> None:
        """Print the environment info"""

        print("="*60)
        print(f"GRAPH ENVIRONMENT INFORMATIONS")
        print("="*60)
        print(f"Graph: {self.graph_name_output} -- {self.graph_name_full}")
        print(f"Community Detection Algorithm: {self.community_detection_alg_name_output}")
        print(f"Number of nodes: {self.original_graph.vcount()}")
        print(f"Number of edges: {self.original_graph.ecount()}")
        print(f"Number of communities: {len(self.original_communities)}")
        #print(f"Communities: {self.original_communities}")
        print("="*60)



