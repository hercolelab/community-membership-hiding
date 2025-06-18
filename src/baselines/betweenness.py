from src.graph_environment.env import GraphEnvironment
import igraph as ig
from typing import List, Tuple, Dict
from dataclasses import dataclass
import copy

@dataclass
class EdgeChange:
    """Class to represent a modification to the graph."""
    edge: Tuple[int, int]
    action_type: str  # 'add' o 'remove'

class CentralityHiding:
    """
    Implementation of the Centrality Hiding baseline.
    The goal is to hide the target node from the target community by rewriting its edges,
    choosing the node with the highest centrality between adding or removing an edge.
    """
    def __init__(
        self, 
        env: GraphEnvironment, 
        target_node: int, 
        budget: int
    ) -> None:
        """
        Initializes the CentralityHiding object.

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
        self.centrality: List[float] = self.env.graph_betweenness
        self._initialize_possible_actions()

    def _initialize_possible_actions(self) -> None:
        """Initializes the list of possible actions ordered by centrality."""
        self.possible_actions: List[Tuple[int, float]] = [
            (node, self.centrality[node])
            for node in range(self.graph.vcount())
            if node != self.target_node
        ]
        # Ordina le azioni per centralità decrescente
        self.possible_actions.sort(key=lambda x: x[1], reverse=True)

    def _apply_edge_change(self, graph: ig.Graph, edge: Tuple[int, int]) -> EdgeChange:
        """
        Applies a modification to the graph (addition or removal of an edge).

        Parameters
        ----------
        graph : ig.Graph
            The graph to modify
        edge : Tuple[int, int]
            The edge to modify

        Returns
        -------
        EdgeChange
            The applied modification
        """
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
        Hides the target node from the target community by rewriting its edges,
        choosing the node with the highest centrality between adding or removing an edge.

        Returns
        -------
        Tuple[ig.Graph, int, Dict[str, List[Tuple[int, int]]]]
            - The graph after applying the baseline
            - The number of steps performed
            - The changes made to the graph
        """
        graph: ig.Graph = self.graph.copy()
        possible_actions: List[Tuple[int, float]] = copy.copy(self.possible_actions)
        changes: Dict[str, List[Tuple[int, int]]] = {"remove": [], "add": []}
        steps: int = 0

        # Loop principale di evasione
        while (self.budget - steps) > 0 and possible_actions:
            # Seleziona il nodo con la centralità più alta
            action: Tuple[int, float] = possible_actions.pop(0)
            edge: Tuple[int, int] = (self.target_node, action[0])

            # Applica la modifica e registra il cambiamento
            edge_change = self._apply_edge_change(graph, edge)
            changes[edge_change.action_type].append(edge_change.edge)
            steps += 1

        return graph, steps, changes