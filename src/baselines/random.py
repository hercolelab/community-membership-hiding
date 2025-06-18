from src.graph_environment.env import GraphEnvironment
from src.utils.utils import ExperimentHyps
import igraph as ig
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random
import copy

@dataclass
class EdgeChange:
    edge: Tuple[int, int]
    action_type: str  # 'add' or 'remove'

class RandomHiding:
    def __init__(
        self, 
        env: GraphEnvironment, 
        target_node: int,
        budget: int, 
    ) -> None:
        """
        Initializes the RandomHiding object.
        The goal is to hide the target node from the target community by rewriting its edges,
        randomly choosing between adding or removing an edge.

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
        self.possible_actions: List[int] = [i for i in range(self.graph.vcount()) if i != self.target_node]

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
        Hides the target node from the target community by rewriting its edges, 
        randomly choosing between adding or removing an edge.

        Returns
        -------
        Tuple[ig.Graph, int, Dict[str, List[Tuple[int, int]]]]
            - The graph after applying the baseline
            - The number of steps performed
            - The changes made to the graph
        """
        graph: ig.Graph = self.graph.copy()
        possible_actions = copy.copy(self.possible_actions)
        changes: Dict[str, List[Tuple[int, int]]] = {"remove": [], "add": []}
        steps: int = 0

        # Fix randomness for reproducibility
        random.seed(ExperimentHyps.seed.value)

        # Evasion attack
        while (self.budget - steps) > 0 and possible_actions:
            action = random.choice(possible_actions)
            edge = (self.target_node, action)
            possible_actions.remove(action)
            edge_change = self._apply_edge_change(graph, edge)
            changes[edge_change.action_type].append(edge_change.edge)
            steps += 1
        return graph, steps, changes