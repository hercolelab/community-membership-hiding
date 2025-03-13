from src.graph_environment.env import GraphEnvironment
from src.utils.utils import ExperimentHyps
import igraph as ig
from typing import List, Tuple
import random
import copy

class RandomHiding():  
    def __init__(
        self, 
        env: GraphEnvironment, 
        target_node: int,
        budget: int, 
    )-> None:
        
        """
        Initialize the RandomHiding object.
        The goal is to hide the target node from the target community by rewiring its edges,
        choosing randomly between adding or removing an edge.

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
        self.possible_actions: List[int] = [i for i in range(self.graph.vcount()) if i != self.target_node]
        
    def community_membership_hiding(self) -> Tuple[ig.Graph, int, dict]:
        """
        Hide the target node from the target community by rewiring its edges, 
        choosing randomly between adding or removing an edge.
        
        Returns
        -------
        graph : ig.Graph
            The graph after the Random Hiding heuristic.
        steps : int
            The number of steps taken to hide the target node
        changes : dict
            The changes made to the graph
        """
        graph: ig.Graph = self.graph.copy()
        possible_actions = copy.copy(self.possible_actions)
        changes: dict = {
            "remove": [],
            "add": [],
        }
        steps: int = 0

        # Fix randomness for reproducibility
        random.seed(ExperimentHyps.seed.value)

        # Evasion attack
        while (self.budget - steps) > 0:         
            
            action = random.choice(possible_actions)
            edge = (self.target_node, action)
            possible_actions.remove(action)
            
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