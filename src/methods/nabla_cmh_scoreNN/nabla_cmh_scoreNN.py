from src.graph_environment.env import GraphEnvironment
from src.community_detection.algorithms import CommunityDetectionAlg
from src.utils.utils import DetectionAlgorithmsNames
from src.methods.nabla_cmh.config import get_hyperparams
from src.methods.nabla_cmh.nabla_utils import nablaUtils
import igraph as ig
from typing import List, Callable, Tuple, Optional, Dict
import numpy as np
import torch
from torch import Tensor, Generator
import torch.optim as optim
from dataclasses import dataclass

@dataclass
class EvasionState:
    """Classe per tenere traccia dello stato dell'evasione"""
    t: int = 0
    budget_used: int = 0
    goal: int = 0
    count_reinit: int = 0
    history: List[Tensor] = None
    edges_changed: Dict = None
    g_prime: ig.Graph = None
    changes: Dict = None
    save_first: bool = False
    x_hat: Tensor = None
    optimizer: torch.optim.Optimizer = None

class nablaCMHv2():
    """
    Versione migliorata del metodo nabla-cmh che utilizza una rete neurale per apprendere le azioni promettenti.
    """

    def __init__(
        self, 
        env: GraphEnvironment, 
        target_node: int,
        budget: int
    ) -> None:
        """
        Inizializza il metodo nabla-cmhv2.

        Parameters
        ----------
        env : GraphEnvironment
            L'ambiente del grafo.
        target_node : int
            Il nodo target da nascondere.
        budget : int
            Il budget di perturbazione disponibile.
        """
        # Configurazione di base
        self.env: GraphEnvironment = env
        self.graph: ig.Graph = self.env.original_graph
        self.budget: int = budget
        self.u: int = target_node
        self.device: torch.device = self.env.device
        self.seed: int = self.env.seed
        self.reinitialization: bool = True
        self.training_alg: str = "greedy"

        # Inizializzazione iperparametri
        self._initialize_hyperparameters()

        # Inizializzazione vettori
        self._initialize_vectors()

        # Variabili per memorizzare una perturbazione valida
        self.last_chance = {}

    def _initialize_hyperparameters(self) -> None:
        """Inizializza gli iperparametri del metodo."""
        self.T, self.lr, self.lambd, self.promising_actions_coeffs = get_hyperparams(
            dataset=self.env.graph_name_output,
            train_alg=self.training_alg,
            tau=self.env.tau,
            beta_factor=self.env.budget_multiplier
        )

    def _initialize_vectors(self) -> None:
        """Inizializza i vettori necessari per il metodo."""
        # Vettore di adiacenza
        self.neighbors: Tensor = torch.LongTensor(self.graph.neighbors(self.u))
        self.a_u: Tensor = torch.zeros(self.graph.vcount(), dtype=torch.int)
        self.a_u[self.neighbors] = 1
        self.a_u = self.a_u.to(self.device)

        # Azioni apprese
        self.a_u_tilde: Tensor = self.learned_actions()
        self.a_u_tilde[self.u] = torch.Tensor([0])
        self.a_u_tilde = self.a_u_tilde.to(self.device)

    def _initialize_evasion_state(self, count_reinit: int = 0) -> EvasionState:
        """
        Inizializza lo stato dell'evasione.

        Parameters
        ----------
        count_reinit : int, optional
            Numero di reinizializzazioni, by default 0

        Returns
        -------
        EvasionState
            Lo stato inizializzato dell'evasione
        """
        x_hat, optimizer = self.initialize_perturbation_vector(count_reinit, self.device)
        return EvasionState(
            history=[self.a_u],
            edges_changed={},
            g_prime=self.graph.copy(),
            changes={"remove": [], "add": []},
            x_hat=x_hat,
            optimizer=optimizer
        )

    def _update_graph(self, state: EvasionState, edges_changed: Dict) -> None:
        """
        Aggiorna il grafo con le modifiche specificate.

        Parameters
        ----------
        state : EvasionState
            Lo stato corrente dell'evasione
        edges_changed : Dict
            Le modifiche da apportare al grafo
        """
        for e in edges_changed["removed"]:
            if state.g_prime.are_connected(*e) or state.g_prime.are_connected(*e[::-1]):
                state.g_prime.delete_edges([e])
        for e in edges_changed["added"]:
            if not state.g_prime.are_connected(*e) and not state.g_prime.are_connected(*e[::-1]):
                state.g_prime.add_edges([e])

    def _check_and_save_first_chance(self, state: EvasionState, da_train: CommunityDetectionAlg) -> None:
        """
        Controlla e salva la prima opportunità di successo.

        Parameters
        ----------
        state : EvasionState
            Lo stato corrente dell'evasione
        da_train : CommunityDetectionAlg
            L'algoritmo di rilevamento delle comunità
        """
        if not state.save_first:
            changes, _ = nablaUtils.get_changes(state.history[0], state.history[-1], self.u)
            self.last_chance = {
                "graph": state.g_prime,
                "budget_used": state.budget_used,
                "changes": changes,
            }

    def community_membership_hiding(self, verbose_iterations: bool = False) -> Tuple[ig.Graph, int, dict, Optional[dict]]:
        """
        Nasconde il nodo target dalla comunità target riscrivendo i suoi archi,
        perturbando il vettore di adiacenza del nodo target usando azioni apprese per guidare la perturbazione.

        Parameters
        ----------
        verbose_iterations : bool, optional
            Se True, la funzione calcola una lista di dizionari che memorizzano informazioni sul processo di ottimizzazione, by default False

        Returns
        -------
        Tuple[ig.Graph, int, dict, Optional[dict]]
            Il grafo dopo l'evasione, il numero di passi, le modifiche apportate e risultati aggiuntivi (se verbose_iterations=True)
        """
        # Inizializzazione
        da_train = CommunityDetectionAlg(self.training_alg, self.env)
        state = self._initialize_evasion_state()
        tp: Tensor = torch.tensor(0.5, device=self.device)
        tn: Tensor = torch.tensor(-0.5, device=self.device)

        # Inizializzazione risultati aggiuntivi se richiesti
        nablaCMH_additional_results = self._initialize_additional_results(verbose_iterations)

        # Loop principale di evasione
        while state.goal == 0 and state.t < self.T:
            # Aggiornamento perturbazione
            p_hat: Tensor = torch.tanh(state.x_hat)
            p: Tensor = nablaUtils.threshold_tanh(p_hat.detach(), tp, tn)
            a_new: Tensor = nablaUtils.clamp(self.a_u + p)
            state.history.append(a_new)
            state.edges_changed, n_changes = nablaUtils.get_changes(state.history[-2], state.history[-1], self.u)

            # Applica modifiche se nel budget
            if n_changes > 0 and (state.budget_used + n_changes <= self.budget):
                state.budget_used += n_changes
                self._update_graph(state, state.edges_changed)
                
                # Verifica obiettivo
                new_communities = da_train.community_detection(state.g_prime)
                new_community_u = self.env.get_community(new_communities)
                state.goal = self.env.get_evasion_goal(new_community_u, None)
                
                # Salva prima opportunità
                self._check_and_save_first_chance(state, da_train)
                
                n_changes = 0  # reset changes

            # Aggiornamento loss e ottimizzazione
            l_decept = self.loss_hide(self.a_u, p_hat, self.a_u_tilde)
            l_dist = self.loss_dist(p_hat)
            loss = l_decept + self.lambd * l_dist
            loss = loss.to(self.device)
            
            state.optimizer.zero_grad()
            loss.backward()
            state.optimizer.step()
            state.t += 1

            # Gestione risultati verbose
            if verbose_iterations:
                self._update_additional_results(nablaCMH_additional_results, state, loss.item())

            # Gestione reinizializzazione
            if state.budget_used > self.budget or (state.budget_used == self.budget and state.goal == 0):
                if self.reinitialization:
                    state = self._handle_reinitialization(state)
                else:
                    break

        # Preparazione risultati finali
        return self._prepare_final_results(state, nablaCMH_additional_results)

    def _initialize_additional_results(self, verbose_iterations: bool) -> Optional[dict]:
        """Inizializza i risultati aggiuntivi se richiesti."""
        if verbose_iterations:
            return {
                "u": self.u,
                "budget": self.budget,
                "count_reinit": 0,
                "iterations": []
            }
        return None

    def _update_additional_results(self, results: dict, state: EvasionState, loss: float) -> None:
        """Aggiorna i risultati aggiuntivi con le informazioni dell'iterazione corrente."""
        results["iterations"].append({
            "t": state.t,
            "loss": loss,
            "Goal": state.goal,
            "Budget used": state.budget_used,
            "Changes": state.edges_changed,
        })

    def _handle_reinitialization(self, state: EvasionState) -> EvasionState:
        """Gestisce la reinizializzazione dello stato."""
        state.count_reinit += 1
        new_state = self._initialize_evasion_state(state.count_reinit)
        new_state.save_first = True
        return new_state

    def _prepare_final_results(self, state: EvasionState, additional_results: Optional[dict]) -> Tuple[ig.Graph, int, dict, Optional[dict]]:
        """Prepara i risultati finali dell'evasione."""
        changes, _ = nablaUtils.get_changes(state.history[0], state.history[-1], self.u)
        
        if state.goal == 0 and state.budget_used < int(self.budget/2):
            if "graph" in self.last_chance:
                return self.last_chance["graph"], self.last_chance["budget_used"], self.last_chance["changes"], additional_results
            return state.g_prime, state.budget_used, changes, additional_results
        
        return state.g_prime, state.budget_used, changes, additional_results

    def loss_hide(self, a_u: Tensor, p_hat: Tensor, a_u_tilde: Tensor) -> float:
        """
        Calcola la loss di nascondimento come distanza tra le azioni promettenti e il vettore di adiacenza perturbato.
        """
        return nablaUtils.frobenius_dist(a_u_tilde, a_u + p_hat) ** 2

    def loss_dist(self, p_hat: Tensor) -> float:
        """
        Calcola la loss di distanza come norma della perturbazione.
        """
        return torch.norm(p_hat)

    def learned_actions(self) -> Tensor:
        """
        Genera il vettore delle azioni apprese per il nodo target u.
        """
        # TODO: Implementare la funzione learned_actions()
        pass

    def initialize_perturbation_vector(self, count_reinit: int, device: torch.device) -> Tuple[Tensor, torch.optim.Optimizer]:
        """
        Inizializza il vettore di perturbazione.
        """
        n_nodes: int = self.graph.vcount()
        gen: Generator = torch.Generator(device=device).manual_seed(self.seed + count_reinit)
        x_hat: Tensor = (2 * torch.rand(n_nodes, device=device, generator=gen) - 1) * 0.5
        x_hat[self.u] = torch.Tensor([0])  # non perturbiamo il nodo target
        x_hat = x_hat.requires_grad_(True)
        optimizer = optim.Adam([x_hat], lr=self.lr)
        return x_hat, optimizer 