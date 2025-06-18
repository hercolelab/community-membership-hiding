# Baselines for Community Membership Hiding

This folder contains baseline algorithms for the **Community Membership Hiding (CMH)** problem. Each script implements a different evasion attack strategy, aiming to hide the membership of a target node in a community by modifying the network structure according to specific heuristics or strategies.

## Contents

- [`dice.py`](./dice.py): Implements the **DICE** attack, which selects the node with the highest degree and rewires its edges to hide community membership, based on a budget.
- [`random.py`](./random.py): Implements the **Random Hiding** attack, which randomly adds or removes edges for the target node, without considering node degree or centrality.
- [`roam.py`](./roam.py): Implements the **ROAM** attack, which modifies the network to decrease the centrality of the target node, aiming to make it less detectable in the community structure.
- [`degree.py`](./degree.py): Implements the **Degree Hiding** attack, which modifies the connections of the target node, prioritizing nodes with higher degree for edge modifications.
- [`betweenness.py`](./betweenness.py): Implements the **Betweenness Hiding** attack, which modifies the connections of the target node, prioritizing nodes with higher betweenness centrality for edge modifications.

## Usage

These scripts are designed to be used as part of the CMH experimental framework. Each file defines a class (e.g., `DiceHiding`, `RandomHiding`, etc.) that can be instantiated with a graph environment, a target node, and a budget. The main method to execute the attack is typically called `community_membership_hiding()`.

## Summary Table

| File              | Method Name         | Description                                                                 |
|-------------------|--------------------|-----------------------------------------------------------------------------|
| `dice.py`         | DICE               | Selects the node with highest degree in/out of the community, and rewires its edges to hide membership. |
| `random.py`       | Random Hiding      | Randomly adds/removes edges for the target node.                            |
| `roam.py`         | ROAM               | Modifies the network to decrease the centrality of the target node.         |
| `degree.py`       | Degree Hiding      | Modifies connections, prioritizing nodes with higher degree.                |
| `betweenness.py`  | Betweenness Hiding | Modifies connections, prioritizing nodes with higher betweenness centrality. |

## Notes
- All methods operate under a budget constraint (maximum number of allowed modifications).
- These baselines provide a reference for evaluating more advanced or learning-based evasion strategies.