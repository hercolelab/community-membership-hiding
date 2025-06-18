from src.graph_environment.env import GraphEnvironment
import igraph as ig
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class EdgeChange:
    edge: Tuple[int, int]
    action_type: str  # 'add' or 'remove'

class RoamHiding:
    def __init__(
            self, 
            env: GraphEnvironment, 
            target_node: int, 
            budget: int,
        ) -> None:
        """
        Initializes the RoamHiding object.
        The goal is to hide the target node from the target community by rewriting its edges,
        in order to reduce the centrality of the target node.
        Inspired by the paper "Hiding Individuals and Communities in a Social Network" by Waniek et al.

        Parameters
        ----------
        env : GraphEnvironment
            The graph environment
        target_node : int
            The target node to hide from the community
        budget : int
            The available perturbation budget
        """
        self.env: GraphEnvironment = env
        self.graph: ig.Graph = self.env.original_graph
        self.budget: int = budget
        self.target_node: int = target_node

    def _apply_edge_change(self, graph: ig.Graph, edge: Tuple[int, int]) -> EdgeChange:
        if graph.are_connected(*edge):
            graph.delete_edges([edge])
            return EdgeChange(edge=edge, action_type="remove")
        elif graph.are_connected(*edge[::-1]):
            graph.delete_edges([edge[::-1]])
            return EdgeChange(edge=edge[::-1], action_type="remove")
        else:
            graph.add_edges([edge])
            return EdgeChange(edge=edge, action_type="add")

    def community_membership_hiding(self) -> Tuple[ig.Graph, int, Dict[str, List[Tuple[int, int]]]]:
        """
        Applies the ROAM heuristic given a budget b:
            - Step 1: Removes the connection between the target node and its chosen neighbor v0
            - Step 2: Connects v0 to b-1 chosen nodes, which are neighbors of the target but not of v0

        Returns
        -------
        Tuple[ig.Graph, int, Dict[str, List[Tuple[int, int]]]]
            - The graph after applying the baseline
            - The number of steps performed
            - The changes made to the graph
        """
        graph: ig.Graph = self.graph.copy()
        changes: Dict[str, List[Tuple[int, int]]] = {"remove": [], "add": []}
        steps: int = 0

        # ---- STEP 1 ---- #
        target_node_neighbours: List[int] = graph.neighbors(self.target_node)
        if len(target_node_neighbours) == 0:
            return graph, steps, changes
        # Choose v0 as the neighbor with the highest degree
        v0: int = max(target_node_neighbours, key=lambda v: graph.degree(v))
        edge: Tuple[int, int] = (self.target_node, v0)
        edge_change = self._apply_edge_change(graph, edge)
        changes[edge_change.action_type].append(edge_change.edge)
        steps += 1

        # ---- STEP 2 ---- #
        v0_neighbours: List[int] = graph.neighbors(v0)
        target_node_neighbours_not_of_v0 = [
            x for x in target_node_neighbours if x not in v0_neighbours and x != v0
        ]
        steps_to_do = min(len(target_node_neighbours_not_of_v0), self.budget - 1)
        sorted_neighbors = sorted(target_node_neighbours_not_of_v0, key=lambda x: graph.degree(x), reverse=True)
        for _ in range(steps_to_do):
            v0_neighbour_of_choice = sorted_neighbors.pop(0)
            edge = (v0, v0_neighbour_of_choice)
            if not graph.are_connected(*edge) and not graph.are_connected(*edge[::-1]):
                edge_change = self._apply_edge_change(graph, edge)
                changes[edge_change.action_type].append(edge_change.edge)
                steps += 1
        return graph, steps, changes    
