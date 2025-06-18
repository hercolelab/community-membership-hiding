# Deep Reinforcement Learning Agent for Community Membership Hiding

This folder contains the implementation of a **Deep Reinforcement Learning (DRL) agent** for the Community Membership Hiding (CMH) problem. The agent is based on the **Advantage Actor-Critic (A2C)** algorithm and is designed to learn optimal strategies for hiding the community membership of a target node by modifying the network structure.

## Contents

- [`agent.py`](./agent.py): Main implementation of the DRL agent. This script defines the environment interface, agent logic, training and evaluation routines, and the interaction loop between the agent and the environment.

- [`a2c/`](./a2c/): Contains the core implementation of the **A2C (Advantage Actor-Critic)** algorithm, including policy and value updates, rollout management, and utilities for parallel training.

- [`models/`](./models/): Contains the neural network architectures used by the agent, such as policy networks and value function approximators. These models are used to process graph data and guide the agent's decisions.

## Approach

The DRL agent leverages the A2C algorithm to learn a policy for edge modifications in the network. The agent observes the current state of the graph, selects actions (additions/removals of edges) to hide the target node's community membership, and receives rewards based on the effectiveness of its actions. Over time, the agent learns to optimize its strategy through trial and error, guided by the reward signal.

Key features:
- **Adaptive learning**: The agent adapts its strategy based on the observed outcomes, enabling it to handle diverse network structures and community detection algorithms.
- **Neural network models**: Used to approximate policy and value functions, allowing the agent to generalize across different graph states.
- **Parallel training**: The A2C implementation supports efficient training using multiple environments in parallel.

## Usage
- The main entry point for training and evaluating the agent is the `agent.py` script/class.
- The `a2c/` subfolder provides the core A2C algorithm and related utilities.
- The `models/` subfolder contains the neural network definitions required by the agent.

## Notes
- This method is intended for advanced, adaptive evasion strategies and can be compared against baseline and optimization-based methods in the CMH framework.
- For detailed usage, configuration, and implementation details, refer to the docstrings and comments within each script and subfolder.

### Advantage Actor-Critic (A2C)

To learn the optimal policy for our agent defined above, we use the **Advantage Actor-Critic** (A2C) algorithm, a popular deep reinforcement learning technique that combines the advantages of both policy-based and value-based methods.
Specifically, A2C defines two neural networks, one for the policy ($\pi_{\theta}$) and another for the value function estimator ($V_v$), such that:

```math
\nabla_{\theta} \mathcal{J} (\theta)  \sim \underset{t=0}{\overset{T-1}{\sum}} \nabla_{\theta} \text{log} \pi_{\theta} (a_t \vert s_t) A(s_t, a_t) 
```
```math
\text{with } A(s_t, a_t) = r_{t+1} + \gamma \mathcal{V}_v(s_{t+1}) - \mathcal{V}_v (s_t)
```
where $\mathcal{J}(\theta)$ is the reward (objective) function, and the goal is to find the optimal policy parameters $\theta$ that maximize it. Instead, $A(s_t, a_t)$ is the advantage function, which quantifies how good or bad an action $a_t$ is compared to the expected value of taking actions according to the current policy.

Below, we describe the policy network (*actor*) and value function network (*critic*) separately.

#### Actor

The policy network is responsible for generating a probability distribution over possible actions based on the input, which consists of a list of nodes and the graph's feature matrix.
However, some graphs may lack node features. In such cases, we can extract continuous node feature vectors (i.e., node embeddings) with graph representational learning frameworks like `node2vec`. These node embeddings serve as the feature matrix.
%ensuring a consistent feature vector size, allowing the model to work with graphs of varying node counts.

Our neural network implementation comprises a primary graph convolution layer (GCNConv) for updating node features. The output of this layer, along with skip connections, feeds into a block consisting of three hidden layers. Each hidden layer includes multi-layer perception (MLP) layers, ReLU activations, and dropout layers. The final output is aggregated using a sum-pooling function. 
The policy is trained to predict the probability that node $v$ is the optimal choice for adding or removing the edge $(u, v)$ to hide the target node $u$ from its original community.
The feasible actions depend on the input node $u$ and are restricted to a subset of the graph's edges. Hence, not all nodes $v \in \mathcal{V}$ are viable options for the policy.

#### Critic

This network closely resembles the one employed for the policy, differing only in one aspect: it incorporates a global sum-pooling operation on the convolution layer's output. This pooling operation results in an output layer with a size of 1, signifying the estimated value of the value function. The role of the value function is to predict the state value when provided with a specific action $a_t$ and state $s_t$

