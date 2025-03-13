from src.graph_environment.env import GraphEnvironment
from src.utils.utils import DetectionAlgorithmsNames, Utils
from src.community_detection.algorithms import CommunityDetectionAlg

import igraph as ig
from typing import List, Callable, Tuple
import random
import copy


class CentralityHiding:
    def __init__(
            self, 
            env: GraphEnvironment, 
            target_node: int, 
            budget: int,
        )-> None:

        """
        Initialize the CentralityHiding object

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
        self.centrality: List[float] = self.env.graph_betweenness
        self.possible_actions: List[Tuple[int,int]] = [
            (node, self.centrality[node])
            for node in range(self.graph.vcount())
            if node != self.target_node
        ]


    def community_membership_hiding(self) -> Tuple[ig.Graph, int, dict]:
        """
        Hide the target node from the target community by rewiring its edges,
        choosing the node with the highest centrality between adding or removing an edge."
        """
        graph: ig.Graph = self.graph.copy()
        possible_actions: List[Tuple[int,int]] = copy.copy(self.possible_actions)
        possible_actions.sort(key=lambda x: x[1], reverse=True)
        changes: dict = { 
            "remove": [],
            "add": [],
        }

        steps: int = 0

        # Evasion attack
        while (self.budget - steps) > 0:  

            # Choose the node with the highest degree
            action: Tuple[int,int] = possible_actions.pop(0) 
            edge: Tuple[int,int] = (self.target_node, action[0])

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