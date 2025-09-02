import torch
from PPOAgent import PPOAgent
from nabla_adapter import NablaAdapter
from methods.nabla_cmh.nabla_cmh import NABLA_CMH
from src.graph_environment.env import GraphEnvironment
from src.utils.utils import FilePaths

def train():
    """
    Funzione principale di training.
    Qui avviene il ciclo di addestramento PPO collegato a NABLA_CMH tramite l'adapter.
    """

    # === Setup dell'ambiente ===
    # In questo esempio scegliamo un grafo (KAR) e un algoritmo di community detection (GRE).
    graph_name = "KAR"
    alg = ["GRE"]
    tau = 0.5         # Soglia di similarità
    c_beta = 1        # Moltiplicatore di budget
    graph_path = getattr(FilePaths, graph_name).value  # Percorso al grafo

    # Creazione dell'ambiente di base (gestisce nodi, grafi, reward "classico")
    env = GraphEnvironment(
        graph_name=graph_name,
        community_detection_algs=alg,
        budget_multiplier=c_beta,
        similarity_threshold=tau,
        graph_path=graph_path,
    )

    # Inizializziamo il metodo NABLA_CMH
    nabla = NABLA_CMH()

    # Creiamo l'adapter che collega PPO ↔ NABLA (traduce azioni in vettori e reward)
    adapter = NablaAdapter(env, nabla)

    # === Setup dell'agente PPO ===
    # Dimensione dello stato (feature del grafo osservato)
    state_dim = env.observation_space.shape[0]
    # Dimensione dell'azione (il nostro "vettore" che NABLA userà)
    action_dim = env.action_space.shape[0]  
    agent = PPOAgent(state_dim, action_dim)

    # === Ciclo di training ===
    num_episodes = 100
    for ep in range(num_episodes):
        # Reset dell'ambiente a inizio episodio
        state = adapter.reset()
        done = False
        total_reward = 0

        while not done:
            # 1. L'agente sceglie un'azione (vettore) a partire dallo stato corrente
            action, log_prob = agent.select_action(state)

            # 2. L'azione viene passata a NABLA tramite l'adapter
            next_state, reward, done, _ = adapter.step(action)

            # 3. Salviamo la transizione nel buffer interno dell'agente
            agent.store_transition(state, action, reward, next_state, done, log_prob)

            # 4. Aggiorniamo lo stato e accumuliamo il reward
            state = next_state
            total_reward += reward

        # Alla fine dell'episodio aggiorniamo la policy PPO con le esperienze accumulate
        agent.update()

        # Log dell'andamento
        print(f"Episode {ep}, Total Reward: {total_reward:.2f}")


# Entry point: lancio del training
if __name__ == "__main__":
    train()