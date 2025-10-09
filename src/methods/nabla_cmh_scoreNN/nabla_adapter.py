import numpy as np
import torch

class NablaAdapter:
    """
    Adapter tra PPO e NABLA_CMH.
    PPO produce un action_vector → NABLA lo interpreta come vettore di punteggi e lo applica sul grafo.
    Questo adapter serve a far comunicare due mondi diversi:
    - l'ambiente RL (GraphEnvironment + PPO),
    - il metodo NABLA_CMH che calcola reward/step su grafi.
    """

    def __init__(self, env, nabla_method):
        """
        Inizializza l'adapter.
        env : istanza di GraphEnvironment (gestisce il grafo, osservazioni, ecc.)
        nabla_method : istanza di NABLA_CMH (algoritmo di selezione nodi su grafo)
        """
        self.env = env
        self.nabla = nabla_method
        self.g_prime = env.original_graph.copy()
        self.budget_used = 0
        self.done = False

    def reset(self):
        """Resetta l'ambiente e restituisce lo stato iniziale (feature nodi)."""
        if hasattr(self.env, "reset"):
            self.env.reset()

        self.g_prime = self.env.original_graph.copy()
        self.budget_used = 0
        self.done = False
        return self.env.get_node_features(self.g_prime)

    def step(self, action_vector):
        if self.done:
            raise RuntimeError("Episodio terminato, chiama reset().")

        # Converti e normalizza il vettore azione
        action_vector = torch.tensor(action_vector, dtype=torch.float32).flatten()
        num_nodes = self.env.original_graph.vcount()
        if action_vector.numel() != num_nodes:
            action_vector = action_vector.repeat(num_nodes)[:num_nodes]
        action_vector = torch.sigmoid(action_vector)

        # Aggiorna lo stato interno di Nabla
        self.nabla.a_u_tilde = action_vector.clone().detach()
        self.nabla.a_u_tilde[self.env.target_node] = 0.0

        # === Esegui UN passo di CMH ===
        g_new, budget_used, changes, _ = self.nabla.community_membership_hiding(verbose_iterations=False)

        # Aggiorna budget e grafo se cambiano
        if g_new is not None:
            self.g_prime = g_new
        self.budget_used += budget_used

        # Calcolo del reward
        new_communities = self.env.nabla_cmh_alg.community_detection(self.g_prime)
        new_community = self.env.get_community(new_communities)
        goal = self.env.get_evasion_goal(new_community, None)

        # Sistemare reward 1.0 perchè rischio troppo piccolo rimane sempre 1 e anche la penalty, giocare con i numeri

        if goal == 1:
            reward = 1.0
            self.done = True
        else:
            old_community = self.env.nabla_cmh_target_community.copy()
            old_community.remove(self.env.target_node)
            overlap = len(set(new_community).intersection(old_community))
            reward = 1.0 - overlap / max(len(old_community), 1)

        # Stato successivo
        features = torch.tensor(self.env.get_node_features(self.g_prime), dtype=torch.float32).flatten()

        # Condizione di terminazione
        if self.budget_used >= self.nabla.budget:
            self.done = True

        info = {
            "budget_used": self.budget_used,
            "changes": changes,
            "goal": goal,
            "graph_edges": self.g_prime.ecount()
        }

        return features, float(reward), self.done, info