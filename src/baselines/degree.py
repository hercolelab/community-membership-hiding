from src.graph_environment.env import GraphEnvironment
from src.utils.utils import DetectionAlgorithmsNames, Utils
from src.community_detection.algorithms import CommunityDetectionAlg

import igraph as ig
from typing import List, Callable, Tuple
import random
import copy


class DegreeHiding:
    def __init__(
            self, 
            env: GraphEnvironment, 
            target_node: int,
            budget: int, 
        )-> None:

        """
        Initialize the DegreeHiding object

        Parameters
        ----------
        env : GraphEnvironment
            The environment object
        target_node : int
            The target node to be hidden from the community
        budget : int
            The budget
        """

        self.env: GraphEnvironment = env
        self.graph: ig.Graph = self.env.original_graph
        self.budget: int = budget
        self.target_node: int = target_node
        self.degrees: List[int] = self.env.graph_degrees
        self.possible_actions: List[Tuple[int,int]] = [
            (node, self.degrees[node]) 
            for node in range(self.graph.vcount()) 
            if node != self.target_node
        ]

    def community_membership_hiding(self) -> Tuple[ig.Graph, List[int], int]:
        """
        Hide the target node from the target community by rewiring its edges,
        choosing the node with the highest degree between adding or removing an edge.

        Returns
        -------
        
        """
        graph: ig.Graph = self.graph.copy()
        possible_actions = copy.copy(self.possible_actions)
        changes: dict = { 
            "remove": [],
            "add": [],
        }

        steps: int = 0
                
        # Evasion attack
        while (self.budget - steps) > 0:  

            # Choose the node with the highest degree
            max_tuple = max(possible_actions, key=lambda x: x[1])
            index = possible_actions.index(max_tuple)
            action = possible_actions.pop(index)
            edge = (self.target_node, action[0])

            if graph.are_connected(*edge):
                graph.delete_edges([edge])
                changes["remove"].append(edge)
            elif graph.are_connected(*edge[::-1]):
                graph.delete_edges([edge[::-1]])
                changes["remove"].append(edge[::-1])
            else:
                graph.add_edges([edge])
                changes["add"].append(edge)
            
            steps += 1
        
        del possible_actions
        return graph, steps, changes