from src.community_detection.algorithms import CommunityDetectionAlg
from src.community_detection.similarity_functions import CommunitySimilarity
from src.utils.utils import Utils, FilePaths, DatasetNames, DatasetFullNames, DetectionAlgorithmsNames, ExperimentHyps, DRL_agentHyps
from typing import List, Tuple, Callable, Optional
import igraph as ig
import numpy as np
import cdlib
import random
import torch
import logging

log = logging.getLogger(__name__)
class GraphEnvironment(object):
    """Class for the Graph Environment"""

    def __init__(
            self, 
            graph_name: str,
            community_detection_algs: List[str],
            target_node: int = 0,
            budget_multiplier: int = 1,
            similarity_threshold: float = 0.5) -> None:
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
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed: int = ExperimentHyps.seed.value

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
        self.target_communities: List[List[int]] = None
        self.target_community_size: int = None
        #self.target_community_sizes: List[int] = None
        self.preferred_community_size: float = ExperimentHyps.target_community_size.value[0]
        self.max_deceptions_for_community: int = ExperimentHyps.max_steps_community_eval.value
        # Budget
        self.budget_multiplier: int = budget_multiplier
        self.budget: int = self.get_budget()
        # Similarity threshold
        self.tau: float = similarity_threshold
        
         # ------ COMMUNITY DETECTION ------ #
        # Community detection algorithm
        self.community_detection_alg_names: List[str] = community_detection_algs
        self.community_detection_alg_names_output: List[str] = [
            getattr(DetectionAlgorithmsNames, alg_name).value for alg_name in self.community_detection_alg_names
        ]
        self.community_detection_algs: List[CommunityDetectionAlg] = None
        self.nabla_cmh_alg: CommunityDetectionAlg = None # Community detection algorithm used by the nabla_cmh method
        # Communities
        self.original_communities: List[cdlib.NodeClustering] = None
        self.old_communities: List[cdlib.NodeClustering] = None # Communities before the last action, used by some methods to compute distances between communities
        self.new_communities: List[cdlib.NodeClustering] = None
        self.nabla_cmh_target_community: List[int] = None # Community used by the nabla_cmh method (also by DICE)
        # Set the community detection algorithm and the communities
        self.set_communities()     

        # ------ SIMILARITY FUNCTIONS ------ #
        # Community Similarity
        self.community_similarity: Callable[[List[int], List[int]], float] = CommunitySimilarity("SOR").select_similarity_function()
        # Graph Similarity
        self.graph_similarity = None

        # ----- GRAPH FEATURES ------ #
        # We compute some features for the graph, which are used by some methods
        self.graph_degrees: List[int] = self.original_graph.degree()
        self.graph_betweenness: List[float] = self.original_graph.betweenness()


        # ------ ENVIRONMENT INFO ------ #
        self.print_environment_info()



    
    # ============================================================================= #
    #                           EPISODE RESET FUNCTIONS                             #
    # ============================================================================= #

    def change_target_community(self) -> None:
        """
        Change the target community according to preferred sizes.
        """
        communities: List[int] = self.original_communities[0].communities
        communities_lenghts: List[int] = [len(c) for c in communities]
        preferred_size: int = int(
            np.ceil(max(communities_lenghts) * self.preferred_community_size)
        )
        closest: int = min(communities_lenghts, key=lambda x: abs(x - preferred_size))
        target_community: List[int] = communities[communities_lenghts.index(closest)].copy()
        self.target_community_size: int = len(target_community)
        random.seed(self.seed)
        if len(target_community) < self.max_deceptions_for_community:
            self.list_target_nodes = target_community
        else: 
            random.shuffle(target_community)
            self.list_target_nodes = target_community[:self.max_deceptions_for_community]

    
    def change_target_node(self, target_node: Optional[int]=None) -> None:
        """
        Change the target node manually or according to community_target_nodes

        Parameters
        ----------
        target_node : Optional[int], default=None
            Target node to be hidden from the community
        """

        if target_node is not None:
            self.target_node = target_node
        else:
            self.target_node = self.list_target_nodes.pop()
        
        # Since we use the community structure of just one algorithm for node sampling
        # we need to update target communities for all algorithms if the target node is not in the previous target community
        for i in range(len(self.target_communities)):
            if self.target_node not in self.target_communities[i]:
                self.target_communities[i] = self.get_community(self.original_communities[i])
        if self.target_node not in self.nabla_cmh_target_community:
            self.nabla_cmh_target_community = self.get_community(self.original_nabla_cmh_communities)


    
    # ============================================================================= #
    #                              SETTERS FUNCTIONS                                #
    # ============================================================================= #


    def set_graph(self) -> None:
        """Set the igraph.Graph object in the environment"""
        
        self.original_graph = Utils.import_graph(getattr(FilePaths, self.graph_name).value)
        self.original_graph = self.set_node_features(self.original_graph)
        #self.graph_agent = self.set_graph_agent()

    def set_node_features(self, graph: ig.Graph) -> None:
        """
        Set the node features in the graph
        """
        for v in graph.vs:
            v["x"] = torch.rand(DRL_agentHyps.EMBEDDING_DIM.value)
        for e in graph.es:
            if "weight" not in e.attributes():
                # Add weight to the edges
                e["weight"] = 1
        return graph

    def set_communities(self) -> None:
        """Set the community detection algorithm class in the environment"""

        self.community_detection_algs = [
            CommunityDetectionAlg(alg_name) for alg_name in self.community_detection_alg_names_output
        ]
        self.nabla_cmh_alg = CommunityDetectionAlg(getattr(DetectionAlgorithmsNames, "GRE").value)
        self.original_communities = [
            alg.community_detection(self.original_graph) for alg in self.community_detection_algs
        ]
        self.original_nabla_cmh_communities = self.nabla_cmh_alg.community_detection(self.original_graph)
        self.old_communities = self.original_communities
        self.target_communities = [
            self.get_community(communities) for communities in self.original_communities
        ]
        self.nabla_cmh_target_community = self.get_community(self.original_nabla_cmh_communities)
        self.target_community_size = len(self.target_communities[0])
        #self.target_community_sizes = [len(community) for community in self.target_communities]
        

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
        if mu <= 2:
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

    def get_community(self, new_community_structure: List[List[int]]) -> List[int]:
        """
        Search the community target in the new community structure for self.target_node. 

        Parameters
        ----------
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
    
    def get_evasion_goal(self, new_community: List[int], alg_idx: Optional[int]) -> int:
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
        if alg_idx is not None:
            old_community_copy = self.target_communities[alg_idx].copy()
        else:
            old_community_copy = self.nabla_cmh_target_community.copy()
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
        
    def get_metrics(self, cf_graph: ig.Graph,) -> Tuple[int,float]:
        
        """
        Compute the goal and NMI metrics.
        """
        goals: List[int] = []
        nmis: List[float] = []
        for idx, alg in enumerate(self.community_detection_algs):
            new_communities: cdlib.NodeClustering = alg.community_detection(cf_graph)
            new_community = self.get_community(new_communities)
            goal: int = self.get_evasion_goal(new_community, idx)
            nmi: float = self.original_communities[idx].normalized_mutual_information(new_communities).score
            goals.append(goal)
            nmis.append(nmi)
        return goals, nmis


    # ============================================================================= #
    #                             ENVIRONMENT INFO                                  #
    # ============================================================================= #

    def print_environment_info(self) -> None:
        """Print the environment info"""

        log.info("="*60)
        log.info("GRAPH ENVIRONMENT INFORMATIONS")
        log.info("="*60)
        log.info(f"Graph: {self.graph_name_output} -- {self.graph_name_full}")
        log.info(f"Community Detection Algorithms: {self.community_detection_alg_names_output}")
        log.info(f"Number of nodes: {self.original_graph.vcount()}")
        log.info(f"Number of edges: {self.original_graph.ecount()}")
        log.info(f"Number of communities: {[len(communities.communities) for communities in self.original_communities]}")
        #log.info(f"Communities: {self.original_communities}")
        log.info("="*60)



