from src.community_detection.algorithms import CommunityDetectionAlg
from src.community_detection.similarity_functions import CommunitySimilarity
from src.utils.utils import Utils, FilePaths, DatasetNames, DatasetFullNames, DetectionAlgorithmsNames, ExperimentHyps
from typing import List, Tuple, Callable, Optional
import igraph as ig
import numpy as np
import cdlib
import random
import time
import torch
import copy
import logging

log = logging.getLogger(__name__)
class GraphEnvironment(object):
    """Class for the Graph Environment"""

    def __init__(
            self, 
            graph_name: str,
            community_detection_alg: str,
            target_node: Optional[int] = 0,
            budget_multiplier: Optional[int] = 1,
            similarity_threshold: Optional[float] = 0.5
    ) -> None:
        """
        Initialize the Graph Environment object

        Parameters
        ----------
        graph_name : str
            The name of the graph, e.g. "KAR"
        community_detection_alg : str
            The name of the community detection algorithm, e,g "GRE"
        target_node : Optional[int], default=None
            The target node to be hidden from the community
        budget_multiplier : Optional[int], default=1
            The budget multiplier
        similarity_threshold : Optional[float], default=0.5
            The similarity threshold   
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

        # ------ COMMUNITY MEMBERSHIP HIDING ------ #
        # Target node
        self.target_node: int = target_node
        self.list_target_nodes: List[int] = None # for the CMH experiment
        # Target community
        self.target_community: List[int] = None
        self.target_community_size: int = None
        self.preferred_community_size: float = ExperimentHyps.target_community_size.value[0]
        self.max_deceptions_for_community: int = ExperimentHyps.max_steps_community_eval.value
        # Budget
        self.budget_multiplier: int = budget_multiplier
        self.budget: int = self.get_budget()
        # Similarity threshold
        self.tau: float = similarity_threshold
        
         # ------ COMMUNITY DETECTION ------ #
        # Community detection algorithm
        self.community_detection_alg_name: str = community_detection_alg
        self.community_detection_alg_name_output: str = getattr(DetectionAlgorithmsNames, self.community_detection_alg_name).value
        self.community_detection_alg: CommunityDetectionAlg = None
        # Communities
        self.original_communities: cdlib.NodeClustering = None
        self.old_communities: cdlib.NodeClustering = None # Communities before the last action, used by some methods to compute distances between communities
        self.new_communities: cdlib.NodeClustering = None
        # Set the community detection algorithm and the communities
        self.set_communities()     

        # ------ SIMILARITY FUNCTIONS ------ #
        # Community Similarity
        self.community_similarity: Callable[[List[int], List[int]], float] = CommunitySimilarity("SOR").select_similarity_function()
        # Graph Similarity
        self.graph_similarity = None

        # ----- GRAPH FEATURES ------ #
        # We compute some features for the graph, which are used by some methods
        self.graph_degrees = self.original_graph.degree()
        self.graph_betweenness = self.original_graph.betweenness()


        # ------ ENVIRONMENT INFO ------ #
        self.print_environment_info()



    
    # ============================================================================= #
    #                           EPISODE RESET FUNCTIONS                             #
    # ============================================================================= #

    def change_target_community(
            self,
            ) -> None:
        """
        Change the target community according to preferred sizes.
        """
        communities: List[int] = self.original_communities.communities
        communities_lenghts: List[int] = [len(c) for c in communities]
        preferred_size: int = int(
            np.ceil(max(communities_lenghts) * self.preferred_community_size)
        )
        closest: int = min(communities_lenghts, key=lambda x: abs(x - preferred_size))
        self.target_community: List[int] = communities[communities_lenghts.index(closest)]
        self.target_community_size: int = len(self.target_community)
        target_community: List[int] = self.target_community.copy()
        random.seed(self.seed)
        random.shuffle(target_community)
        self.list_target_nodes = target_community[:self.max_deceptions_for_community]

    
    def change_target_node(
            self,
            step: int,
            target_node: Optional[int]=None,
        ) -> None:
        """
        Change the target node manually or according to community_target_nodes

        Parameters
        ----------
        step : int
            Step of the episode, i.e. index in the list
        target_node : Optional[int], default=None
            Target node to be hidden from the community
        """

        if target_node is not None:
            self.target_node = target_node
        else:
            self.target_node = self.list_target_nodes[step]

    
    # ============================================================================= #
    #                              SETTERS FUNCTIONS                                #
    # ============================================================================= #


    def set_graph(self) -> None:
        """Set the igraph.Graph object in the environment"""
        
        self.original_graph = Utils.import_graph(getattr(FilePaths, self.graph_name).value)
        #self.graph_agent = self.set_graph_agent()

    def set_communities(self) -> None:
        """Set the community detection algorithm class in the environment"""

        self.community_detection_alg = CommunityDetectionAlg(self.community_detection_alg_name_output)
        self.original_communities = self.community_detection_alg.community_detection(self.original_graph)
        self.old_communities = self.original_communities
        self.target_community = self.get_community(self.original_communities)
        self.target_community_size = len(self.target_community)
        

    # ============================================================================= #
    #                              GETTERS FUNCTIONS                                #
    # ============================================================================= #

    def get_average_budget(self) -> int:
        """
        Get the average budget for the graph, which is equal to |E|/|V|.
        We consider just the integer part of this division.
        If the division result is less than 3, we consider |E|/|V| + 1.
        """
        mu = self.original_graph.ecount() // self.original_graph.vcount()
        if mu < 2:
            return mu + 1
        return mu
    
    def get_budget(self) -> int:
        """
        Get the budget for the environment based on the budget multiplier and average budget
        
        Returns
        -------
        int
            The budget for the environment
        """
        budget: int = int(self.get_average_budget() * self.budget_multiplier)
        return budget

    def get_community(
        self,
        new_community_structure: List[List[int]]
    ) -> List[int]:
        """
        Search the community target in the new community structure after changes. 

        Parameters
        ----------
        node_target : int
            Target node to be hidden from the community
        new_community_structure : List[List[int]]
            New community structure after deception

        Returns
        -------
        List[int]
            New community target after deception
        """
        for community in new_community_structure.communities:
            if self.target_node in community:
                return community
        raise ValueError("Community not found")
    
    def get_evasion_goal(self, new_community: List[int]) -> int:
        """
        Check if the goal of hiding the target node was achieved

        Parameters
        ----------
        new_community : int
            New community of the target node

        Returns
        -------
        int
            1 if the goal was achieved, 0 otherwise
        """
        # If the target node is the only node in the community, the goal is achieved
        if len(new_community) == 1:
            return 1
        # Copy the communities to avoid modifying the original ones
        new_community_copy = new_community.copy()
        new_community_copy.remove(self.target_node)
        old_community_copy = self.target_community.copy()
        old_community_copy.remove(self.target_node)
        # Compute the similarity between the new and the old community
        similarity = self.community_similarity(
            new_community_copy,
            old_community_copy
        )
        del new_community_copy, old_community_copy
        if similarity <= self.tau:
            return 1
        return 0
        
    def get_metrics(
        self,
        cf_graph: ig.Graph,
    ) -> Tuple[int,float]:
        
        """
        Compute the goal and NMI metrics.
        """
        new_communities: cdlib.NodeClustering = self.community_detection_alg.community_detection(cf_graph)
        new_community = self.get_community(new_communities)
        goal: int = self.get_evasion_goal(new_community)
        nmi: float = self.original_communities.normalized_mutual_information(new_communities).score
        return goal, nmi


    # ============================================================================= #
    #                             ENVIRONMENT INFO                                  #
    # ============================================================================= #

    def print_environment_info(self) -> None:
        """Print the environment info"""

        log.info("="*60)
        log.info("GRAPH ENVIRONMENT INFORMATIONS")
        log.info("="*60)
        log.info(f"Graph: {self.graph_name_output} -- {self.graph_name_full}")
        log.info(f"Community Detection Algorithm: {self.community_detection_alg_name_output}")
        log.info(f"Number of nodes: {self.original_graph.vcount()}")
        log.info(f"Number of edges: {self.original_graph.ecount()}")
        log.info(f"Number of communities: {len(self.original_communities.communities)}")
        #log.info(f"Communities: {self.original_communities}")
        log.info("="*60)



