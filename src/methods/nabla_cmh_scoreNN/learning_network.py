import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LearningNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LearningNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Per outputtare valori in [0,1]

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return self.sigmoid(x)  # Output un vettore di score in [0,1]

def backpropagate(model, optimizer, state, action, reward, next_state, done):
    """
    Esegue la backpropagation per l'addestramento della rete neurale.
    
    Parameters
    ----------
    model : LearningNetwork
        Il modello della rete neurale.
    optimizer : torch.optim.Optimizer
        L'ottimizzatore per l'addestramento.
    state : torch.Tensor
        Lo stato corrente.
    action : torch.Tensor
        L'azione intrapresa.
    reward : float
        La ricompensa ottenuta.
    next_state : torch.Tensor
        Il prossimo stato.
    done : bool
        Indica se l'episodio è terminato.
    """
    # Calcola la loss/reward
    loss = compute_loss(model, state, action, reward, next_state, done)
    
    # Backpropagation e aggiornamento dei pesi
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train(model, optimizer, env, nabla_cmh, epochs, time_step):
    """
    Esegue il training della rete neurale per un numero di epoche.
    
    Parameters
    ----------
    model : LearningNetwork
        Il modello della rete neurale.
    optimizer : torch.optim.Optimizer
        L'ottimizzatore per l'addestramento.
    env : GraphEnvironment
        L'ambiente del grafo.
    nabla_cmh : nablaCMH
        Il metodo nabla-cmh.
    epochs : int
        Il numero di epoche per il training.
    time_step : int
        Il numero di epoche dopo cui cambiare il nodo target.
    """
    for epoch in range(epochs):
        # Esegui una run del metodo nabla-cmh
        new_graph, steps, changes, _ = nabla_cmh.community_membership_hiding()
        
        # Estrai lo stato, l'azione e la ricompensa
        state = env.get_state()  # Implementare questa funzione
        action = env.get_action()  # Implementare questa funzione
        reward = env.get_reward()  # Implementare questa funzione
        next_state = env.get_next_state()  # Implementare questa funzione
        done = env.is_done()  # Implementare questa funzione
        
        # Esegui la backpropagation
        loss = backpropagate(model, optimizer, state, action, reward, next_state, done)
        
        # Cambia il nodo target ogni time_step epoche
        if (epoch + 1) % time_step == 0:
            env.change_target_node()

def compute_loss(model, state, action, reward, next_state, done):
    """
    Calcola la loss per l'addestramento.
    La loss è basata sulla differenza tra l'output della rete (score predetto) 
    e l'azione effettivamente intrapresa, pesata dalla ricompensa.
    
    Parameters
    ----------
    model : LearningNetwork
        Il modello della rete neurale.
    state : torch.Tensor
        Lo stato corrente.
    action : torch.Tensor
        L'azione intrapresa (vettore di 0 e 1).
    reward : float
        La ricompensa ottenuta.
    next_state : torch.Tensor
        Il prossimo stato.
    done : bool
        Indica se l'episodio è terminato.
    
    Returns
    -------
    loss : torch.Tensor
        La loss calcolata.
    """
    # Predici gli score per ogni possibile azione
    predicted_scores = model(state)
    
    # Calcola la loss come MSE tra gli score predetti e l'azione effettiva,
    # pesata dalla ricompensa
    loss = F.mse_loss(predicted_scores, action) * reward
    
    return loss 