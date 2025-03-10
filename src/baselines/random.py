from src.graph_environment.env import GraphEnvironment
from src.utils.utils import DetectionAlgorithmsNames, Utils, ExperimentHyps
from src.community_detection.algorithms import CommunityDetectionAlg
import igraph as ig
from typing import List, Callable, Tuple
import random
import time
import copy

class RandomHiding():
    """
    Baseline for CMH where we modify the target node's neighbourhood randomly
    """
    
    def __init__(
        self, 
        env: GraphEnvironment, 
        target_node: int,
        budget: int, 
    )-> None:
        
        self.env: GraphEnvironment = env
        self.graph: ig.Graph = self.env.original_graph
        self.budget: int = budget
        self.target_node: int = target_node
        self.possible_edges: List[Tuple[int,int]] = self.get_possible_actions()

    def get_possible_actions(self):
        # Put all edges between the target node and its neighbors in a list
        neighbors: set = set(self.graph.neighbors(self.target_node))
        possible_actions_remove: List[Tuple[int,int]] = [(self.target_node, neighbor) for neighbor in neighbors]
        
        # Put all the edges that aren't neighbors of the target node in a list
        possible_actions_add: List[Tuple[int,int]] = [(self.target_node, node.index) for node in self.graph.vs if node.index != self.target_node and node.index not in neighbors]
        
        possible_actions: List[Tuple[int,int]] = possible_actions_add + possible_actions_remove
        return possible_actions
        
    def hide_target_node_from_community(self) -> Tuple[ig.Graph, List[int], int]:
        """
        Hide the target node from the target community by rewiring its edges, 
        choosing randomly between adding or removing an edge.
        
        Returns
        -------
        Tuple[ig.Graph, List[int], int]
            The new graph, the new community structure and the number of steps
        """
        graph: ig.Graph = self.graph.copy()
        possible_edges = copy.copy(self.possible_edges)
        changes: dict = {
            "remove": [],
            "add": [],
        }
        steps: int = 0

        # Fix randomness for reproducibility
        random.seed(ExperimentHyps.seed.value)

        # Evasion attack
        while (self.budget - steps) > 0:         
            
            edge = random.choice(possible_edges)
            possible_edges.remove(edge)
            
            if graph.are_connected(*edge):
                graph.remove_edges(*edge)
                changes["remove"].append(edge)
            elif graph.are_connected(*edge[::-1]):
                graph.remove_edges(*edge[::-1])
                changes["remove"].append(edge[::-1])
            else:
                graph.add_edge(*edge)
                changes["add"].append(edge)
            
            steps += 1
        del possible_edges
        return graph, steps, changes