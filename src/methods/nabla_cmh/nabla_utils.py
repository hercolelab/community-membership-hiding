from src.graph_environment.env import GraphEnvironment
from typing import List, Tuple
import igraph as ig
import torch
from torch import Tensor
import numpy as np
import cdlib
import copy
from scipy.stats import rankdata



class nablaUtils:
    """Class to store utility functions"""

    # ============================================================================= #
    #                               PROMISING ACTIONS                               #
    # ============================================================================= #

    @staticmethod
    def compute_promising_scores(env: GraphEnvironment, att_coeffs: List[float]) ->Tensor:
        """
        Compute the scores related to promising actions.
        We consider the following graph features:
            - Betweenness centrality
            - Degree
            - Intra-community degree
            - Inter-community degree
        The scores are computed as follows:
            1. Compute the ranks of the features
            2. Scale the ranks between 0 and 1
            3. Aggregate the scores with the attention coefficients
        
        Parameters
        ----------
        env : GraphEnvironment
            Graph environment
        att_coeff : List[float]
            Attention coefficients

        
        Returns
        -------
        att : Tensor
            Tensor containing the aggregated scores.
        """

        communities: cdlib.NodeClustering = copy.deepcopy(env.original_communities)
        graph: ig.Graph = env.original_graph.copy()
        target_community = env.target_community.copy()

        # ---- Centrality score ---- #

        centrality: np.ndarray = np.array(env.graph_betweenness)
        #Compute ranking scores
        ranks: np.ndarray = rankdata(centrality, method='average')
        #Normalize ranks
        att1: np.ndarray = (ranks - 1) / (len(ranks) - 1)

        
        # ---- Degree score ---- #

        degrees: np.ndarray = np.array(env.graph_degrees)
        #Compute ranking scores
        ranks: np.ndarray = rankdata(degrees, method='average')
        #Normalize ranks
        att2: np.ndarray = (ranks - 1) / (len(ranks) - 1)


        # ---- Intra-community degree score ---- #
        
        att3a: np.ndarray = np.zeros(graph.vcount())
        for c in communities.communities:
            if len(c) > 1:
                subgraph = graph.subgraph(c)
                sub_degrees = np.array(subgraph.degree())
                sub_ranks = rankdata(sub_degrees, method='average')
                sub_norm_ranks = (sub_ranks - 1) / (len(sub_ranks) - 1) 
                att3a[c] = sub_norm_ranks
            
        # ---- Inter-community degree score ---- #

        att3b = np.zeros(graph.vcount())
        inter_comm_nodes = sorted(set(range(graph.vcount())) - set(target_community))
        if len(inter_comm_nodes) > 2:
            subgraph = graph.subgraph(inter_comm_nodes)
            sub_degrees = np.array(subgraph.degree())
            sub_ranks = rankdata(sub_degrees, method='average')
            sub_norm_ranks = (sub_ranks - 1) / (len(sub_ranks) - 1) 
            att3b[inter_comm_nodes] = sub_norm_ranks

        # ---- Aggregate scores ---- #

        if len(att_coeffs) != 4:
            raise ValueError("Attention coefficients have not the same lenght of the considered features")
        att = att_coeffs[0]*att1 + att_coeffs[1]*att2 + att_coeffs[2]*att3a + att_coeffs[3]*att3b
        
        return Tensor(att)

    # ============================================================================= #
    #                               GRAPH OPERATIONS                                #
    # ============================================================================= #

    @staticmethod
    def get_changes(v_before: Tensor, v_after: Tensor, target_node: int) -> Tuple[dict, int]:
        """
        Get the changes in the target_node adjacency vector.
        We suppose this function works on undirected graphs.

        Parameters
        ----------
        v_before : Tensor
            Adjacency vector before the changes.
        v_after : Tensor
            Adjacency vector after the changes.
        target_node
            Target node.

        Returns
        -------
        changes
            Dictionary containing the changes in the adjacency vector.
            The structure is the following:
                {
                    "added": [(u,v), ...],
                    "removed": [(u,v), ...]
                }
        n_changes
            Number of changes.
        """

        diff: Tensor = v_before != v_after
        indices: Tensor = torch.nonzero(diff).flatten()
        changes: dict = {"added": [], "removed": []}
        n_changes: int = len(indices)
        u: int = target_node

        for i in indices.tolist():
            edge = (u,i)
            if v_before[i] == 1:
                changes["removed"].append(edge)
            else:
                changes["added"].append(edge)
        
        return changes, n_changes
    
    @staticmethod
    def update_edge_list(edge_list: List[Tuple[int,int]], changes: dict) -> List[Tuple[int,int]]:
        """
        Update edge list according to the changes in the graph.

        Parameters
        ----------
        edge_list
            List of edges in the graph.

        changes
            Dictionary containing the changes in the graph.

        Returns
        -------
        updated_edge_list
            Updated list of edges in the graph.
        """

        edge_set = set(edge_list)
        removed_set = set(changes["removed"])
        added_set = set(changes["added"])
        edge_set.difference_update(removed_set)
        edge_set.update(added_set)
        updated_edge_list = list(edge_set)
        return updated_edge_list


    # ============================================================================= #
    #                               TORCH OPERATIONS                                #
    # ============================================================================= #

    @staticmethod
    def clamp(z: Tensor) -> Tensor:
        """
        Clamp the tensor between 0 and 1, according to the equation clamp(z) = max(0, min(1, z)).

        Parameters
        ----------
        z : Tensor
            Tensor to clamp
        device : str
            Device to use
        
        Returns
        -------
        z_out: Tensor
            Clamped tensor
        """
        z_out = torch.clamp(z,min=0,max=1)
        return z_out
    
    @staticmethod
    def threshold_tanh(z: Tensor, tp: float, tn: float) -> Tensor:
        """
        Threshold the tensor into {-1,0,1}, according to the equation 
            threshold_tanh(z) = -1 if z <= tn, 0 if tn < z < tp, 1 if z >= tp.

        Parameters
        ----------
        z : Tensor
            Tensor to threshold
        tp : float
            Positive threshold
        tn : float
            Negative threshold
        
        Returns
        -------
        z_out: Tensor
            Thresholded Tensor.

        """
        z_out = torch.zeros_like(z, device=z.device, dtype=torch.int)
        z_out[z >= tp] = 1
        z_out[z <= tn] = -1
        return z_out
    
    def frobenius_dist(v1: Tensor, v2: Tensor) -> Tensor:
        """
        Frobenius norm between two vectors.

        Parameters
        ----------
        v1 : Tensor
            First vector
        v2 : Tensor
            Second vector
        """
        return torch.norm(v1 - v2)

