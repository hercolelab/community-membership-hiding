import numpy as np

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
        state = self.env.reset()
        return np.array(state, dtype=np.float32)

    def step(self, action_vector):
        """
        Step adattato:
        - PPO genera un action_vector (il nostro "vettore di punteggi")
        - Questo viene passato a NABLA, che esegue la sua logica sul grafo.
        - NABLA restituisce: next_state, reward, done, info
        """
        next_state, reward, done, info = self.nabla.step(self.env, action_vector)
        return np.array(next_state, dtype=np.float32), reward, done, info