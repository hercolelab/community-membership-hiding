import datetime
import os
import json
import torch
import matplotlib.pyplot as plt
from src.methods.nabla_cmh_scoreNN.PPOAgent import PPOAgent
from src.methods.nabla_cmh_scoreNN.nabla_adapter import NablaAdapter
from src.methods.nabla_cmh.nabla_cmh import nablaCMH
from src.graph_environment.env import GraphEnvironment
from src.utils.utils import FilePaths


def plot_and_save(results, save_dir, prefix=""):
    """
    Salva i grafici dei risultati. 
    Se prefix è impostato, aggiunge il prefisso al nome dei file, utile per checkpoint intermedi.
    Genera grafici separati per reward, steps, loss e done.
    """
    episodes = range(len(results["reward"]))
    metrics = ["reward", "steps", "loss", "done"]
    titles = ["Reward", "Steps", "Loss", "Done (0=forzato, 1=normale)"]

    for metric, title in zip(metrics, titles):
        plt.figure()
        plt.plot(episodes, results[metric])
        plt.xlabel("Episode")
        plt.ylabel(title)
        plt.title(f"{title} per episodio")
        plt.grid(True)
        filename = f"{prefix}_{metric}.png" if prefix else f"{metric}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()


def save_checkpoint(agent, episode, save_dir):
    """Salva il checkpoint del modello e dell'ottimizzatore."""
    ckpt_path = os.path.join(save_dir, f"checkpoint_ep{episode}.pth")
    torch.save({
        'episode': episode,
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }, ckpt_path)
    print(f"Checkpoint salvato all'episodio {episode} in {ckpt_path}")
    return ckpt_path


def load_checkpoint(agent, checkpoint_path):
    """
    Carica un checkpoint esistente.
    - checkpoint_path: percorso completo al file .pth salvato
    Restituisce l'episodio da cui ripartire.
    """
    checkpoint = torch.load(checkpoint_path)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint caricato dall'episodio {checkpoint['episode']}")
    return checkpoint['episode']


def train(resume_checkpoint=None):
    """
    Funzione principale di training PPO collegato a NABLA_CMH tramite l'adapter.
    
    Parameters
    ----------
    resume_checkpoint : str, default=None
        Percorso completo del checkpoint da cui riprendere. Esempio:
        "results/ppo_training_20250922_120000/checkpoint_ep200.pth"
        Se None, il training parte da zero.
    """

    # --- Cartella dei risultati ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    SAVE_DIR = os.path.join(BASE_DIR, "results", f"ppo_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Parametri dell'ambiente ---
    graph_name = "KAR"
    alg = ["GRE"]
    tau = 0.5
    c_beta = 1
    graph_path = getattr(FilePaths, graph_name).value

    # --- Inizializzazione ambiente e NABLA ---
    env = GraphEnvironment(
        graph_name=graph_name,
        community_detection_algs=alg,
        budget_multiplier=c_beta,
        similarity_threshold=tau,
        graph_path=graph_path,
    )
    nabla = nablaCMH(env, target_node=5, budget=0.5) # Inserire numero nodo (target_node) e budget (budget) da voler trainare
    adapter = NablaAdapter(env, nabla)

    # --- Setup agente PPO ---
    state_dim = env.original_graph.vcount()  # numero di nodi = dimensione dello stato
    action_dim = env.original_graph.vcount() # numero di nodi = dimensione dell'azione
    agent = PPOAgent(state_dim, action_dim)

    # --- Caricamento checkpoint se presente ---
    start_episode = 0
    if resume_checkpoint:
        start_episode = load_checkpoint(agent, resume_checkpoint)

    # --- Parametri training ---
    num_episodes = 500
    MAX_STEPS = 500

    rewards_log, steps_log, losses_log, done_log = [], [], [], []

    # --- Ciclo principale di training ---
    for ep in range(start_episode, num_episodes):
        state = adapter.reset()
        done = False
        forced_done = False
        total_reward = 0
        step_count = 0

        while not done:
            # --- Selezione azione e interazione con l'ambiente ---
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = adapter.step(action)
            agent.store_transition(state, action, reward, next_state, done, log_prob)

            state = next_state
            total_reward += reward
            step_count += 1

            # --- Termina forzatamente se raggiunto MAX_STEPS ---
            if step_count >= MAX_STEPS:
                forced_done = True
                done = True

        # --- Aggiornamento PPO dopo l'episodio ---
        loss = agent.update()
        losses_log.append(loss if loss is not None else 0.0)
        rewards_log.append(total_reward)
        steps_log.append(step_count)
        done_log.append(0 if forced_done else 1)

        print(f"[Ep {ep}] Reward={total_reward:.2f}, Steps={step_count}, Loss={loss:.4f}, Done={done_log[-1]}")

        # --- Salvataggio checkpoint e grafico intermedio ogni 100 episodi ---
        if (ep + 1) % 100 == 0:
            ckpt_path = save_checkpoint(agent, ep+1, SAVE_DIR)
            results = {"reward": rewards_log, "steps": steps_log, "loss": losses_log, "done": done_log}
            plot_and_save(results, SAVE_DIR, prefix=f"ep{ep+1}")

    # --- Salvataggi finali ---
    final_path = os.path.join(SAVE_DIR, "ppo_final.pth")
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }, final_path)

    results = {"reward": rewards_log, "steps": steps_log, "loss": losses_log, "done": done_log}
    with open(os.path.join(SAVE_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Grafici finali ---
    plot_and_save(results, SAVE_DIR)

    print("Training completato. Risultati salvati in:", SAVE_DIR)


if __name__ == "__main__":
    train()