from src.graph_environment.env import GraphEnvironment
import igraph as ig
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

@dataclass
class EdgeChange:
    edge: Tuple[int, int]
    action_type: str  # 'add' or 'remove'

class DiceHiding:
    def __init__(
            self,
            env: GraphEnvironment,
            target_node: int,
            budget: int,
            dice_coefficient: Optional[float] = 0.5
        ) -> None:
        """
        Initializes the DiceHiding object.
        The goal is to hide the target node from the target community by rewriting its edges,
        following the DICE heuristic (Disconnect Internally, Connect Externally).
        Inspired by the paper "Hiding Individuals and Communities in a Social Network" by Waniek et al.

        Parameters
        ----------
        env : GraphEnvironment
            The graph environment
        target_node : int
            The target node to hide from the community
        budget : int
            The available perturbation budget
        dice_coefficient : float, optional
            DICE coefficient (default 0.5)
        """
        self.env: GraphEnvironment = env
        self.graph: ig.Graph = self.env.original_graph
        self.budget: int = budget
        self.target_node: int = target_node
        self.target_community: List[int] = env.nabla_cmh_target_community
        self.dice_coefficient: float = dice_coefficient

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
        Applies the DICE heuristic given a budget b:
            - Step 1: Disconnects d <= b edges between the target node and its neighbors in the community
            - Step 2: Connects the target node to b-d nodes outside the community

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

        intra_changes: int = 1  # d changes
        extra_changes: int = self.budget - intra_changes
        if (intra_changes + extra_changes) != self.budget:
            raise ValueError("The sum of intra_changes and extra_changes must be equal to the budget")

        # ---- STEP 1 ---- #
        subgraph: ig.Graph = graph.induced_subgraph(self.target_community)
        new_id_target_node = self.target_community.index(self.target_node)
        intra_neighbours: List[int] = subgraph.neighbors(new_id_target_node)
        if len(intra_neighbours) < intra_changes:
            intra_changes = len(intra_neighbours)
            extra_changes = self.budget - intra_changes
        if len(intra_neighbours) == 0:
            raise ValueError("The target node has no neighbours in the target community")
        sorted_intra_neighbours: List[int] = sorted(intra_neighbours, key=lambda x: subgraph.degree(x), reverse=True)
        for _ in range(intra_changes):
            intra_neighbour_of_choice: int = self.target_community[sorted_intra_neighbours.pop(0)]
            edge: Tuple[int, int] = (self.target_node, intra_neighbour_of_choice)
            edge_change = self._apply_edge_change(graph, edge)
            changes[edge_change.action_type].append(edge_change.edge)
            steps += 1

        # ---- STEP 2 ---- #
        extra_nodes: List[int] = [i for i in range(graph.vcount()) if i not in self.target_community and i not in self.graph.neighbors(self.target_node)]
        if len(extra_nodes) == 0:
            while (len(sorted_intra_neighbours) > 0 and steps < self.budget):
                intra_neighbour_of_choice: int = self.target_community[sorted_intra_neighbours.pop(0)]
                edge: Tuple[int, int] = (self.target_node, intra_neighbour_of_choice)
                edge_change = self._apply_edge_change(graph, edge)
                changes[edge_change.action_type].append(edge_change.edge)
                steps += 1
            return graph, steps, changes

        sorted_extra_nodes: List[int] = sorted(extra_nodes, key=lambda x: graph.degree(x), reverse=True)
        for _ in range(extra_changes):
            if not sorted_extra_nodes:
                break
            extra_node_of_choice: int = sorted_extra_nodes.pop(0)
            edge: Tuple[int, int] = (self.target_node, extra_node_of_choice)
            if not graph.are_connected(*edge) and not graph.are_connected(*edge[::-1]):
                edge_change = self._apply_edge_change(graph, edge)
                changes[edge_change.action_type].append(edge_change.edge)
                steps += 1
        return graph, steps, changes
