import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ValuePolicy(nn.Module):
    """
    Rete unica che produce:
      - policy (mean e std per la distribuzione gaussiana delle azioni)
      - value (valutazione dello stato)
    L'output principale rimane l'action_vector.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Strati condivisi
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Policy head → distribuzione azioni
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  

        # Value head → V(s)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Forward unico: produce mean, std e value.
        La policy e il value sono separati solo "logicamente",
        ma condividono le prime due fully connected.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        mean = self.mean(x)
        std = torch.exp(self.log_std)  # std = exp(log_std) per garantire positività
        value = self.value(x)

        return mean, std, value


class PPOAgent:
    """
    Implementazione dell'agente PPO.
    - Tiene un buffer interno delle transizioni (rollout)
    - Seleziona azioni da PolicyNetwork
    - Calcola valori con ValueNetwork
    - Aggiorna policy e value secondo l'algoritmo PPO
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, update_steps=10):
        self.model = ValuePolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_steps = update_steps

        # Buffer interno (evitiamo file separati)
        self.reset_buffer()

    def reset_buffer(self):
        """Svuota il buffer di transizioni (nuovo episodio)"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    # ========================================================= #
    #                 CICLO DI INTERAZIONE RL                   #
    # ========================================================= #
    def select_action(self, state):
        """
        Dato lo stato → campiona un'azione dalla distribuzione della policy.
        Restituisce sia il vettore di azioni che la sua log-probabilità.
        """
        state = torch.tensor(state, dtype=torch.float32)
        mean, std, _ = self.model(state)
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
        Calcola i ritorni (discounted returns) dell'episodio.
        Usa il valore stimato del next_state come bootstrap.
        """
        returns = []
        R = 0
        next_state = torch.tensor(next_state, dtype=torch.float32)
        # bootstrap con il valore stimato dello stato finale
        _, _, value_next = self.model(next_state)
        R = value_next.item()
        # scorri all'indietro le ricompense
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns


    # ========================================================= #
    #                    AGGIORNAMENTO PPO                      #
    # ========================================================= #
    def update(self):
        """
        Aggiornamento PPO dopo aver raccolto un episodio.
        - Calcola returns e advantage
        - Applica clipping del ratio
        - Update policy + value
        - Restituisce la loss media fatta sui passi di update (utile per logging/plot)
        """
        # Convertiamo buffer in tensori
        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.float32)
        old_log_probs = torch.stack(self.log_probs).detach()
        returns = torch.tensor(
            self.compute_returns(self.rewards, self.dones, self.next_states[-1]),
            dtype=torch.float32
        )

        losses = []
        for _ in range(self.update_steps):
            # Forward: policy + value insieme
            mean, std, values = self.model(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1)

            # Advantage: differenza tra return e valore stimato
            advantages = returns - values.detach().squeeze()

            # PPO ratio & clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            value_loss = (returns - values.squeeze()).pow(2).mean()

            # Loss totale
            loss = policy_loss + 0.5 * value_loss

            # Ottimizzazione
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        # Pulisci buffer per prossimo episodio
        self.reset_buffer()

        # Calcola e restituisci la loss media sugli update steps
        mean_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
        return mean_loss