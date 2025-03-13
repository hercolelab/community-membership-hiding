from src.graph_environment.env import GraphEnvironment
import igraph as ig
from typing import List, Tuple
import copy


class DegreeHiding:
    def __init__(
            self, 
            env: GraphEnvironment, 
            target_node: int,
            budget: int, 
        )-> None:

        """
        Initialize the DegreeHiding object.
        The goal is to hide the target node from the target community by rewiring its edges,
        choosing the node with the highest degree between adding or removing an edge.

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

    def community_membership_hiding(self) -> Tuple[ig.Graph, int, dict]:
        """
        Hide the target node from the target community by rewiring its edges,
        choosing the node with the highest degree between adding or removing an edge.

        Returns
        -------
        graph : ig.Graph
            The graph after the Degree Hiding heuristic.
        steps : int
            The number of steps taken to hide the target node
        changes : dict
            The changes made to the graph
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