from src.graph_environment.env import GraphEnvironment
import igraph as ig
from typing import Tuple, List, Optional

class DiceHiding:
    def __init__(
            self,
            env: GraphEnvironment,
            target_node: int,
            budget: int,
            dice_coefficient: Optional[float] = 0.5
        ) -> None:
    
        """
        Initialize the DiceHiding object.
        The goal is to hide the target node from the target community by rewiring its edges,
        where the heuristic is "Disconnect Internally, Connect Externally" (DICE).
        Inspired from the article "Hiding Individuals and Communities in a Social Network" by Waniek et al.

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
        self.target_community: List[int] = env.target_community
        self.dice_coefficient: float = dice_coefficient
    

    def community_membership_hiding(self) -> Tuple[ig.Graph, int, dict]:
        """
        The DICE heuristic given a budget b:
            - Step 1: Disconnect d <= b links between the target node and its neighbours 
            of choice within the community
            - Step 2: Connect the target node to d-b nodes of choice, who are not in the community
        
        Returns
        -------
        graph : ig.Graph
            The graph after the DICE heuristic.
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

        intra_changes: int = int(self.budget * self.dice_coefficient) # d changes
        extra_changes: int = self.budget - intra_changes
        if (intra_changes + extra_changes) != self.budget:
            raise ValueError("The sum of intra_changes and extra_changes must be equal to the budget")
        
        # ---- STEP 1 ---- #
        
        # Create a subgraph containing only the target community
        subgraph: ig.Graph = graph.subgraph(self.target_community)
        # Get neighbours of the target node in the subgraph
        new_id_target_node = self.target_community.index(self.target_node)
        intra_neighbours: List[int] = subgraph.neighbors(new_id_target_node)
        if len(intra_neighbours) < intra_changes:
            intra_changes: int = len(intra_neighbours)
            extra_changes: int = self.budget - intra_changes
        if len(intra_neighbours) == 0:
            raise ValueError("The target node has no neighbours in the target community")
        # We remove intra_changes edges between the target node and its neighbours
        # Our choice is to disconnect the target node from the neighbour with the highest intra_degree
        sorted_intra_neighbours: List[int] = sorted(intra_neighbours, key=lambda x: subgraph.degree(x), reverse=True)
        for _ in range(intra_changes):
            intra_neighbour_of_choice: int = self.target_community[sorted_intra_neighbours.pop(0)]
            edge: Tuple[int,int] = (self.target_node, intra_neighbour_of_choice)
            if graph.are_connected(*edge):
                graph.delete_edges([edge])
                changes["remove"].append(edge)
            elif graph.are_connected(*edge[::-1]):
                graph.delete_edges([edge[::-1]])
                changes["remove"].append(edge[::-1])
            steps += 1

        # ---- STEP 2 ---- #

        # Get the nodes not in the target community and not neighbours of the target node
        extra_nodes: List[int] = [i for i in range(graph.vcount()) if i not in self.target_community and i not in self.graph.neighbors(self.target_node)]
        if len(extra_nodes) == 0:
            raise ValueError("There are no nodes outside the target community")
        sorted_extra_nodes: List[int] = sorted(extra_nodes, key=lambda x: graph.degree(x), reverse=True)
        for _ in range(extra_changes):
            extra_node_of_choice: int = sorted_extra_nodes.pop(0)
            edge: Tuple[int,int] = (self.target_node, extra_node_of_choice)
            if not graph.are_connected(*edge) and not graph.are_connected(*edge[::-1]):
                graph.add_edges([edge])
                changes["add"].append(edge)
                steps += 1
        return graph, steps, changes
