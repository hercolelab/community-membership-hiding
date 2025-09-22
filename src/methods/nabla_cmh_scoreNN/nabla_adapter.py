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

    def reset(self):
        """
        Reset dell'ambiente di base.
        Restituisce lo stato iniziale in formato numpy.float32
        """
        # Generiamo uno stato dummy, ad esempio un vettore zero
        state = np.zeros(self.env.original_graph.vcount(), dtype=np.float32)
        return state

    def step(self, action_vector):
        """
        Step adattato:
        - PPO genera un action_vector
        - Viene simulata la logica di NABLA senza modificare il file nabla
        """
        # Applichiamo il vettore come perturbazione iniziale
        self.nabla.a_u = torch.tensor(action_vector, dtype=torch.float32, device=self.nabla.device)

        # Chiamata alla funzione principale di CMH
        g_prime, budget_used, changes, _ = self.nabla.community_membership_hiding(verbose_iterations=False)

        # Stato successivo dummy
        next_state = np.zeros(self.env.original_graph.vcount(), dtype=np.float32)

        # Reward semplice
        reward = float(len(changes.get("removed", [])) - len(changes.get("added", [])))

        # Done se budget superato o goal raggiunto
        done = budget_used >= self.nabla.budget

        info = {"changes": changes}

        return next_state, reward, done, info