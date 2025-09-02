import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """
    Policy network PPO.
    Data un'osservazione (stato), produce la distribuzione gaussiana
    da cui campionare il vettore di azioni.
    - Output: media e deviazione standard della distribuzione Normal
    - Azione: vettore di punteggi (uno per dimensione)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Parametri della distribuzione delle azioni
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # std ottimizzabile

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)  # std positivo
        return mean, std


class ValueNetwork(nn.Module):
    """
    Critico di PPO: stima il valore atteso dello stato (V(s)).
    Serve a calcolare gli advantage e stabilizzare l'update.
    """
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.value(x)


class PPOAgent:
    """
    Implementazione dell'agente PPO.
    - Tiene un buffer interno delle transizioni (rollout)
    - Seleziona azioni da PolicyNetwork
    - Calcola valori con ValueNetwork
    - Aggiorna policy e value secondo l'algoritmo PPO
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, update_steps=10):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_steps = update_steps

        # Buffer interno (evitiamo file separati)
        self.reset_buffer()

    def reset_buffer(self):
        """Svuota il buffer di transizioni (nuovo episodio)"""
        self.states, self.actions, self.rewards, self.next_states, self.dones, self.log_probs = [], [], [], [], [], []

    def select_action(self, state):
        """
        Dato lo stato → campiona un'azione dalla distribuzione della policy.
        Restituisce sia il vettore di azioni che la sua log-probabilità.
        """
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()  # il "vettore" che sarà passato a NABLA
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob.detach()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Salva una transizione nel buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def compute_returns(self, rewards, dones, next_state):
        """
        Calcola i ritorni (reward cumulativi scontati).
        Usa anche il value del next_state come bootstrap.
        """
        returns, R = [], 0
        next_state = torch.tensor(next_state, dtype=torch.float32)
        R = self.value(next_state).item()  # bootstrap sul prossimo stato
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def update(self):
        """
        Aggiornamento PPO.
        - Calcola advantage
        - Applica clipping ratio
        - Aggiorna policy + value con gradiente
        """
        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.float32)
        old_log_probs = torch.stack(self.log_probs).detach()
        returns = torch.tensor(
            self.compute_returns(self.rewards, self.dones, self.next_states[-1]),
            dtype=torch.float32
        )

        for _ in range(self.update_steps):
            # Distribuzione attuale
            mean, std = self.policy(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1)

            # Valori stimati
            values = self.value(states).squeeze()
            advantages = returns - values.detach()

            # PPO clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            value_loss = (returns - values).pow(2).mean()

            # Ottimizzazione
            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Puliamo buffer per nuovo episodio
        self.reset_buffer()