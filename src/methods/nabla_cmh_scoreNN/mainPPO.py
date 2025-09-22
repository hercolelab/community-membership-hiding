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


def plot_and_save(results, save_dir):
    """
    Funzione di supporto per plottare e salvare i risultati del training.
    Genera i grafici di reward, steps, loss e done.
    """
    episodes = range(len(results["reward"]))

    # === Reward ===
    plt.figure()
    plt.plot(episodes, results["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per episodio")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "reward.png"))
    plt.close()

    # === Steps ===
    plt.figure()
    plt.plot(episodes, results["steps"])
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per episodio")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "steps.png"))
    plt.close()

    # === Loss ===
    plt.figure()
    plt.plot(episodes, results["loss"])
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss per episodio")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    # === Done ===
    plt.figure()
    plt.plot(episodes, results["done"])
    plt.xlabel("Episode")
    plt.ylabel("Done (0=forzato, 1=normale)")
    plt.title("Done per episodio")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "done.png"))
    plt.close()


def train():
    """
    Funzione principale di training.
    Qui avviene il ciclo di addestramento PPO collegato a NABLA_CMH tramite l'adapter.
    """

    # === Setup cartella risultati divisa con data e ora ===
    ### TIMESTAMP RUN (puoi rimuoverla se vuoi sovrascrivere ogni volta)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory di questo file
    SAVE_DIR = os.path.join(BASE_DIR, "results", f"ppo_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # === Setup cartella risultati sovrascribile ===
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    # SAVE_DIR = os.path.join(BASE_DIR, "results", "ppo_training")  
    # os.makedirs(SAVE_DIR, exist_ok=True)

    # === Setup dell'ambiente ===
    graph_name = "KAR"
    alg = ["GRE"]
    tau = 0.5
    c_beta = 1
    graph_path = getattr(FilePaths, graph_name).value

    env = GraphEnvironment(
        graph_name=graph_name,
        community_detection_algs=alg,
        budget_multiplier=c_beta,
        similarity_threshold=tau,
        graph_path=graph_path,
    )

    # Inizializziamo NABLA
    nabla = nablaCMH(env,target_node=5,budget=0.5)

    # Adapter PPO ↔ NABLA
    adapter = NablaAdapter(env, nabla)

    # === Setup agente PPO ===
    state_dim = env.original_graph.vcount()  # numero di nodi = dimensione dello stato
    action_dim = env.original_graph.vcount() # numero di nodi = dimensione dell'azione
    agent = PPOAgent(state_dim, action_dim)

    # === Ciclo di training ===
    num_episodes = 1000
    MAX_STEPS = 200

    rewards_log, steps_log, losses_log, done_log = [], [], [], []

    for ep in range(num_episodes):
        state = adapter.reset()
        done = False
        forced_done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = adapter.step(action)
            agent.store_transition(state, action, reward, next_state, done, log_prob)

            state = next_state
            total_reward += reward
            step_count += 1

            if step_count >= MAX_STEPS:
                print(f"[Ep {ep}] MAX_STEPS={MAX_STEPS} raggiunto, termino forzatamente.")
                forced_done = True
                done = True

        loss = agent.update()
        losses_log.append(loss if loss is not None else 0.0)
        rewards_log.append(total_reward)
        steps_log.append(step_count)
        done_log.append(0 if forced_done else 1)

        print(f"[Ep {ep}] Reward={total_reward:.2f}, Steps={step_count}, Loss={loss:.4f}, Done={done_log[-1]}")

        if (ep + 1) % 200 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_ep{ep+1}.pth")
            torch.save({
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict()
            }, ckpt_path)

    # === Salvataggi finali ===
    final_path = os.path.join(SAVE_DIR, "ppo_final.pth")
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }, final_path)

    results = {"reward": rewards_log, "steps": steps_log, "loss": losses_log, "done": done_log}
    with open(os.path.join(SAVE_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_and_save(results, SAVE_DIR)
    print("Training completato. Risultati salvati in:", SAVE_DIR)


if __name__ == "__main__":
    train()