# Methods for Community Membership Hiding

This folder contains advanced methods for solving the **Community Membership Hiding (CMH)** problem. Each subfolder implements a different approach, ranging from deep reinforcement learning to optimization-based strategies.

## Contents

- [`drl_agent/`](./drl_agent/): Implements a **Deep Reinforcement Learning (DRL) agent** based on the Advantage Actor-Critic (A2C) algorithm. The agent learns to perform edge modifications in the network to optimally hide the membership of a target node within a community. The approach leverages neural networks to approximate policy and value functions, enabling adaptive and data-driven evasion strategies.

- [`nabla_cmh/`](./nabla_cmh/): Implements the **nabla-CMH** method, an optimization-based approach for community membership hiding. This method formulates the evasion problem as a constrained optimization task, seeking the minimal set of edge modifications required to hide the target node's community membership while respecting a given budget and similarity constraints.

- [`nabla_cmhv2/`](./nabla_cmhv2/): Extension of the nabla-CMH method. *(Description to be added.)*

## Notes
- These methods are designed to provide advanced, adaptive, or optimal solutions to the CMH problem, and can be compared against the baselines for performance evaluation.
- For implementation details, refer to the docstrings and comments within each subfolder. 