from src.graph_environment.env import GraphEnvironment
from src.community_detection.algorithms import CommunityDetectionAlg
from src.utils.utils import DetectionAlgorithmsNames
from src.methods.nabla_cmh.config import get_hyperparams
from src.methods.nabla_cmh.nabla_utils import nablaUtils
import igraph as ig
from typing import List, Callable, Tuple, Optional
import numpy as np
import torch
from torch import Tensor, Generator
import torch.optim as optim

class nablaCMH():
    def __init__(
        self, 
        env: GraphEnvironment, 
        target_node: int,
        budget: int
    ) -> None:
        self.env: GraphEnvironment = env
        self.graph: ig.Graph = self.env.original_graph
        self.budget: int = budget
        self.u: int = target_node
        self.device: torch.device = self.env.device
        self.seed: int = self.env.seed
        self.reinitialization: bool = True # to choose if allow the method to reinit optimization if goal not achieved
        self.training_alg: str = "greedy"

        # Hyperparameters
        self.T, self.lr, self.lambd, self.promising_actions_coeffs = get_hyperparams(
            dataset=self.env.graph_name_output,
            train_alg=self.training_alg,
            tau=self.env.tau,
            beta_factor=self.env.budget_multiplier
        )

        # Adjacency vector
        self.neighbors: Tensor = torch.LongTensor(self.graph.neighbors(self.u))
        self.a_u: Tensor = torch.zeros(self.graph.vcount(), dtype=torch.int)
        self.a_u[self.neighbors] = 1

        # Promising actions
        self.a_u_tilde: Tensor = self.promising_actions()
        self.a_u_tilde[self.u] = torch.Tensor([0])
        self.a_u_tilde = self.a_u_tilde.to(self.device)

        # Variables to store a valid perturbation
        # This is returned if in the end there are no changes
        self.last_chance = {}


    
    # ============================================================================= #
    #                                MAIN FUNCTION                                  #
    # ============================================================================= #

    def community_membership_hiding(self, verbose_iterations: bool=False) -> Tuple[ig.Graph, int, dict, Optional[dict]]:
        """
        Hide the target node from the target community by rewiring its edges,
        perturbing the adjacency vector of the target node using promising actions to guide the perturbation.

        Parameters
        ----------
        verbose_iterations: bool
            If it is true, then the function compute a list of dictionaries
            which store informations about the optimizatio process
        Returns
        -------
        graph : ig.Graph
            The graph after the Degree Hiding heuristic.
        steps : int
            The number of steps taken to hide the target node
        changes : dict
            The changes made to the graph
        """

        # Set the training detection algorithm
        da_train = CommunityDetectionAlg(self.training_alg, self.env)
        # Evasion parameters
        t: int = 0
        budget_used: int = 0
        goal: int = 0
        count_reinit: int = 0
        self.a_u = self.a_u.to(self.device)
        history: List[Tensor] = [self.a_u]
        edges_changed: dict = {}
        # Counterfactual graph
        g_prime: ig.Graph = self.graph.copy()
        changes: dict = { 
            "remove": [],
            "add": [],
        }
        save_first = False
        #Perturbation vector
        #x_hat, optimizer = self.initialize_perturbation_vector(count_reinit, self.device)
        p_hat = torch.zeros_like(self.a_u, dtype=torch.float, requires_grad=True, device=self.device)
        optimizer = optim.Adam([p_hat], lr=self.lr)
        eps = 1e-4
        logit_init = torch.logit(self.a_u.float().clamp(eps, 1-eps))
        #tp: Tensor = torch.tensor(0.5, device=self.device)
        #tn: Tensor = torch.tensor(-0.5, device=self.device)

        if verbose_iterations:
            nablaCMH_additional_results = {
                "u": self.u,
                "budget": self.budget,
                "count_reinit": 0,
                "iterations": []
            }
        else:
            nablaCMH_additional_results = None

        list_changes = []

        # ---- Evasion Loop ---- #

        while goal==0 and t < self.T:
            
            #Perturbation update
            #p_hat: Tensor = torch.tanh(x_hat)
            #p: Tensor = nablaUtils.threshold_tanh(p_hat.detach(),tp,tn)
            h = torch.sigmoid(logit_init + p_hat)
            #a_new: Tensor = nablaUtils.clamp(self.a_u + p)

            a_new = (h > 0.5).int()
            history.append(a_new)

            #l_decept = self.loss_hide(self.a_u, p_hat, self.a_u_tilde)
            #l_dist = self.loss_dist(p_hat)
            #loss = l_decept + self.lambd * l_dist
            loss = torch.nn.functional.binary_cross_entropy(h,self.a_u_tilde, reduction='mean')
            loss = loss.to(self.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t += 1
            
            edges_changed, n_changes = nablaUtils.get_changes(history[-2], history[-1], self.u)

            if n_changes > 0:
                budget_used += n_changes
                #edge_list = g_prime.get_edgelist() # inefficient
                #updated_edge_list = nablaUtils.update_edge_list(edge_list,edges_changed) # inefficient
                for e in edges_changed["removed"]:
                    if g_prime.are_connected(*e) or g_prime.are_connected(*e[::-1]):
                        g_prime.delete_edges([e])
                        list_changes.append(e)
                for e in edges_changed["added"]:
                    if not g_prime.are_connected(*e) and not g_prime.are_connected(*e[::-1]):
                        g_prime.add_edges([e])
                        list_changes.append(e)
                new_communities = da_train.community_detection(g_prime)
                new_community_u = self.env.get_community(new_communities)
                goal = self.env.get_evasion_goal(new_community_u, None)
                n_changes = 0 #reset changes
                
            
            if budget_used > self.budget:
                g_prime = self.graph.copy()
                top_beta_changes = {
                    "remove": [],
                    "add": []
                }
                diff_indices = torch.where(self.a_u != a_new)[0]
                diff_scores = torch.abs(h[diff_indices] - self.a_u[diff_indices].float())
                top_indices = diff_indices[torch.argsort(diff_scores, descending=True)[:self.budget]]

                for idx in top_indices:
                    if self.a_u[idx] == 1:
                        e = (self.u, idx.item())
                        top_beta_changes["remove"].append(e)
                        if g_prime.are_connected(*e) or g_prime.are_connected(*e[::-1]):
                            g_prime.delete_edges([e])
                    else:
                        e = (self.u, idx.item())
                        top_beta_changes["add"].append(e)
                        if not g_prime.are_connected(*e) and not g_prime.are_connected(*e[::-1]):
                            g_prime.add_edges([e])                

                return g_prime, self.budget, top_beta_changes, nablaCMH_additional_results
            
        changes, _ = nablaUtils.get_changes(history[0], history[-1], self.u)
        return g_prime, budget_used, changes, nablaCMH_additional_results
                
        
        


    # ============================================================================= #
    #                                  LOSS FUNCTIONS                               #
    # ============================================================================= #
    
    def loss_hide(self, a_u: Tensor, p_hat: Tensor, a_u_tilde: Tensor) -> float:
        """
        Compute the hide loss as the distance between the promising actions and the perturbed adjacency vector.
        
        Parameters
        ----------
        a_u: Tensor
            The original adjacency vector of the target node
        p_hat: Tensor
            The continuos perturbation
        a_u_tilde: Tensor
            The promising actions vector
            
        Returns
        -------
        l_hide: Float
            The value of the deception loss.
        """
        l_hide = nablaUtils.frobenius_dist(a_u_tilde,a_u+p_hat)**2
        return l_hide
    
    def loss_dist(self, p_hat: Tensor):
        """Compute the distance loss as the norm of the perturbation
        
        Parameters
        ----------
        p: Tensor
            Perturbation vector (continuous)
            
        Returns
        -------
        l_dist: float
            The value of the distance loss.
        """
        l_dist = torch.norm(p_hat)
        return l_dist
    

    # ============================================================================= #
    #                                 UTILS FUNCTIONS                               #
    # ============================================================================= #


    def promising_actions(self) -> Tensor:
        """
        Generate the promising actions vector for the target node u.

        Returns
        -------
        prom_actions : torch.Tensor
            Tensor containing the promising actions.
        """

        n = self.env.original_graph.vcount()
        L = torch.ones(n)
        L_in = torch.LongTensor(self.env.nabla_cmh_target_community)
        L[L_in] = torch.Tensor([0])
        scores = torch.Tensor(nablaUtils.compute_promising_scores(self.env,self.promising_actions_coeffs))
        #prom_actions = torch.where(L == 1, 0.5 + scores/2, 0.5 - scores/2)
        prom_actions = self.a_u + scores * (L - self.a_u)
        return prom_actions
    

    def initialize_perturbation_vector(self, count_reinit: int, device: torch.device) -> Tuple[Tensor, torch.optim.Optimizer]:
        """
        Initialize the perturbation vector s.t. threshold(tanh(x_hat)) = 0,
        namely we generate a random vector in [-0.5,0.5].

        Parameters
        ----------
        count_reinit : int
            The number of reinitializations.
        device : torch.device
            The device to use.
        
        Returns
        -------
        x_hat
            The perturbation vector.
        optimizer
            The optimizer.
        """

        n_nodes: int = self.graph.vcount()
        gen: Generator = torch.Generator(device=device).manual_seed(self.seed + count_reinit)
        x_hat: Tensor = (2*torch.rand(n_nodes, device=device, generator=gen) - 1)*0.5
        x_hat[self.u] = torch.Tensor([0]) # we do not perturb the target node
        x_hat = x_hat.requires_grad_(True)
        optimizer = optim.Adam([x_hat], lr=self.lr)
        return x_hat,optimizer