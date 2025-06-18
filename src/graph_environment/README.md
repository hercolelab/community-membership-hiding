# Graph Environment

This folder contains the environment definition for the **Community Membership Hiding (CMH)** problem.

## File: `env.py`

### Purpose

`env.py` implements the core environment class and related utilities used throughout the CMH framework. The environment is responsible for managing the graph structure, community information, and the interaction logic required by various evasion algorithms (baselines, optimization methods, and reinforcement learning agents).

### Main Responsibilities
- **Graph Management**: Loads and maintains the network structure, including nodes, edges, and their attributes.
- **Community Detection**: Interfaces with community detection algorithms to identify and update community structures within the graph.
- **Target Selection**: Handles the selection and management of target communities and target nodes for the hiding task.
- **Budget and Constraints**: Manages the budget for edge modifications and enforces similarity or other constraints during the evasion process.
- **Metrics and Evaluation**: Provides methods to compute evaluation metrics (e.g., goal achievement, NMI) for the effectiveness of evasion strategies.
- **State and Action Interface**: Defines the state representation and action space for use by reinforcement learning agents and other algorithms.

### Key Classes and Functions
- **`GraphEnvironment`**: The main class that encapsulates all environment logic, including:
  - Initialization and loading of graphs and communities
  - Methods for changing the target community and node
  - Methods for applying edge modifications (additions/removals)
  - Budget and constraint management
  - Metric computation and logging
  - State and action interfaces for agents
- **Utility Functions**: Additional helper functions for graph manipulation, community analysis, and environment resets.

### Usage
- The `GraphEnvironment` class is instantiated by all major modules in the CMH framework, including baselines, optimization-based methods (e.g., nabla-CMH), and reinforcement learning agents (e.g., A2C agent).
- It provides a unified interface for interacting with the graph and managing the evasion process, ensuring consistency across different methods.

### Notes
- For detailed usage and implementation, refer to the docstrings and comments within `env.py`.
- The environment is designed to be flexible and extensible, supporting a variety of graph types, community detection algorithms, and evasion strategies. 