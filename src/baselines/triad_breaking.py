from src.graph_environment.env import GraphEnvironment
import igraph as ig
from typing import List, Tuple, Dict
from dataclasses import dataclass
import copy

@dataclass
class EdgeChange:
    edge: Tuple[int, int]
    action_type: str  # 'add' or 'remove'

class TriadBreakingHiding:
    """
    Baseline that hides the target node by breaking the largest number of triangles (triads)
    it is involved in. It removes edges that destroy the highest number of local closed triads.
    """

    def __init__(self, env: GraphEnvironment, target_node: int, budget: int) -> None:
        self.env: GraphEnvironment = env
        self.graph: ig.Graph = self.env.original_graph
        self.target_node: int = target_node
        self.budget: int = budget
        self.triad_scores: Dict[Tuple[int, int], int] = {}  # edge -> num_triads_broken

        self._compute_edge_triad_scores()

    def _get_triangles_with_target(self) -> List[Tuple[int, int, int]]:
        """Returns all triangles that involve the target node."""
        triangles = self.graph.cliques(min=3, max=3)
        return [tuple(sorted(tri)) for tri in triangles if self.target_node in tri]

    def _compute_edge_triad_scores(self) -> None:
        """For each edge involving the target node, count how many triangles it participates in."""
        triangles = self._get_triangles_with_target()
        score_dict: Dict[Tuple[int, int], int] = {}

        for tri in triangles:
            u, v, w = tri
            other_nodes = [n for n in tri if n != self.target_node]
            for node in other_nodes:
                edge = tuple(sorted((self.target_node, node)))
                score_dict[edge] = score_dict.get(edge, 0) + 1

        self.triad_scores = score_dict

    def _apply_edge_removal(self, graph: ig.Graph, edge: Tuple[int, int]) -> EdgeChange:
        if graph.are_connected(*edge):
            graph.delete_edges([edge])
            return EdgeChange(edge=edge, action_type="remove")
        elif graph.are_connected(*edge[::-1]):
            graph.delete_edges([edge[::-1]])
            return EdgeChange(edge=edge[::-1], action_type="remove")
        else:
            return None

    def community_membership_hiding(self) -> Tuple[ig.Graph, int, Dict[str, List[Tuple[int, int]]]]:
        graph = self.graph.copy()
        changes: Dict[str, List[Tuple[int, int]]] = {"remove": [], "add": []}
        steps = 0

        # Ordina le edge candidate per numero di triadi spezzate (decrescente)
        sorted_edges = sorted(self.triad_scores.items(), key=lambda x: x[1], reverse=True)

        for edge, score in sorted_edges:
            if steps >= self.budget:
                break
            if graph.are_connected(*edge):
                change = self._apply_edge_removal(graph, edge)
                if change:
                    changes["remove"].append(change.edge)
                    steps += 1

        return graph, steps, changes