from src.graph_environment.env import GraphEnvironment
import igraph as ig
from typing import List, Tuple
import copy


class RoamHiding:
    def __init__(
            self, 
            env: GraphEnvironment, 
            target_node: int, 
            budget: int,
        ) -> None:

        """
        Initialize the RoamHiding object.
        The goal is to hide the target node from the target community by rewiring its edges,
        so that the centrality of the target node is reduced.
        From the article "Hiding Individuals and Communities in a Social Network" by Waniek et al.

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

    def community_membership_hiding(self) -> Tuple[ig.Graph, int, dict]:
        """
        The ROAM heuristic given a budget b:
            - Step 1: Remove the link between the source node, v, and its
            neighbour of choice, v0;
            - Step 2: Connect v0 to b-1 nodes of choice, who are neighbours
            of v but not of v0 (if there are fewer than b-1 such neighbours,
            connect v0 to all of them).

        Returns
        -------
        graph : ig.Graph
            The graph after the ROAM heuristic.
        steps : int
            The number of steps taken to hide the target node
        changes : dict
            The changes made to the graph
        """
        graph: ig.Graph = self.graph.copy()
        changes: dict = { 
            "remove": [],
            "add": [],
        }
        steps: int = 0

        # ---- STEP 1 ---- #

        target_node_neighbours : list = graph.neighbors(self.target_node)
        if len(target_node_neighbours) == 0:
            #print("No neighbours for the target node", self.target_node)
            return graph, steps, changes
        # Choose v0 as the neighbour of target_node with higher degree (our choice), or randomly
        v0: int = target_node_neighbours[0]
        for v in target_node_neighbours:
            if graph.degree(v) > graph.degree(v0):
                v0 = v
        # Remove the edge between the target node and v0
        edge: Tuple[int,int] = (self.target_node, v0)
        if graph.are_connected(*edge):
                graph.delete_edges([edge])
                changes["remove"].append(edge)
        elif graph.are_connected(*edge[::-1]):
            graph.delete_edges([edge[::-1]])
            changes["remove"].append(edge[::-1])
        steps += 1

        # ---- STEP 2 ---- #

        v0_neighbours: list = graph.neighbors(v0)
        target_node_neighbours_not_of_v0 = [
            x for x in target_node_neighbours if x not in v0_neighbours and x != v0
        ]
        # If there are fewer than b-1 such neighbours, connect v_0 to all of them
        if len(target_node_neighbours_not_of_v0) < self.budget - 1:
            steps_to_do = len(target_node_neighbours_not_of_v0)
        else:
            steps_to_do = self.budget - 1

        # We connect v_0 to steps_to_do nodes of choice, who are neighbours of v but not of v_0
        # Our choice is to connect v_0 to the nodes with higher degree
        sorted_neighbors = sorted(target_node_neighbours_not_of_v0, key=lambda x: graph.degree(x), reverse=True)
        while (steps_to_do > 0):
            v0_neighbour_of_choice = sorted_neighbors.pop(0)
            edge = (v0, v0_neighbour_of_choice)
            if not graph.are_connected(*edge) and not graph.are_connected(*edge[::-1]):
                graph.add_edges([edge])
                changes["add"].append(edge)
                steps += 1
            steps_to_do -= 1
        return graph, steps, changes    
