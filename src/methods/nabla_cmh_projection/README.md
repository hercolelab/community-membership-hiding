# nabla-CMH Method

This folder contains the implementation of the **nabla-CMH** method for the Community Membership Hiding (CMH) problem. The nabla-CMH approach formulates the evasion problem as a constrained optimization task, aiming to find the minimal set of edge modifications required to hide the community membership of a target node, while respecting budget and similarity constraints.

## Contents

- [`nabla_cmh.py`](./nabla_cmh.py): Main implementation of the nabla-CMH algorithm. This script defines the core logic for the optimization-based evasion attack, including the main class and methods for running the attack, evaluating solutions, and interacting with the graph environment.

- [`nabla_utils.py`](./nabla_utils.py): Utility functions and helper classes used by nabla-CMH. This includes graph manipulation routines, similarity computations, and other supporting tools required for the optimization process.

- [`config.py`](./config.py): Configuration file containing hyperparameters, default settings, and utility functions for managing experiment configurations. This file centralizes the control of parameters used by nabla-CMH and related scripts.

- [`nabla_cmh-hyp_search.py`](./nabla_cmh-hyp_search.py): Script for hyperparameter search and tuning. This script allows systematic exploration of different parameter settings to optimize the performance of the nabla-CMH method.

## Overview

The nabla-CMH method is designed to:
- Identify a set of edge modifications (additions/removals) that hide the target node's community membership.
- Minimize the number of modifications, subject to a budget constraint.
- Preserve the overall structure and similarity of the original network.

## Usage
- The main entry point for running the attack is typically the `nabla_cmh.py` script/class.
- Utility functions in `nabla_utils.py` are used internally by the main algorithm.
- Experiment and algorithm parameters can be set or modified in `config.py`.
- For hyperparameter optimization, use `nabla_cmh-hyp_search.py`.

## Notes
- For detailed usage and implementation, refer to the docstrings and comments within each script.
- This method can be compared against other baselines and advanced methods for performance evaluation in the CMH framework. 

## In-depth analysis of the method
The community membership problem is inherently discrete, rendering it unsuitable for direct optimisation using gradient-based techniques.
To overcome this, we adopt a strategy by introducing a *perturbation vector* $p$ that is applied to the adjacency vector $A_u$: 

$$ 
A'_u = clamp(A_u + p),
$$

where $p \in \{-1,0,1\}^{|V|}$. Intuitively, a value of $-1$ in $p$ corresponds to removing an existing edge or leaving a non-existent edge unaltered, $0$ preserves the current edge state, and $1$ either adds a new edge or retains an existing one. The function $clamp(x) = \max (0, \min (x,1))$ ensures that the elements of new adjacency vector $A_u'$ are contained to $\{0,1\}$, mapping the set $\{-1,0,1,2\}$ to binary values.
However, since the values in $p$ remain discrete, we first introduce a real-valued vector $\hat{p}$, whose entries are constrained to the range $[-1,1]$ using a $\tanh$ transformation. These values are then thresholded to obtain the discrete perturbation vector $p$, defined as:

- $p_{i} =+1  \quad \text{if } \hat{p}_{i} \geq t^+$
- $p_{i} = -1  \quad \text{if } \hat{p}_{i} \leq t^- $
- $p_{i} = 0  \quad \text{otherwise.}$


A straightforward choice for the thresholds is $t^+=0.5$ and $t^-=-0.5$, which cleanly separates positive, negative, and neutral perturbations. 
Consequently, $\hat{p}$ becomes the \emph{only} set of parameters subject to optimisation, fully governing the perturbation process.

The optimisation operates on a 
vector $\hat{p}$ initialised uniformly within $[-0.5, 0.5]^{|V|}$, which corresponds to starting the process from a null perturbation state. 
Since we assume no internal knowledge of $f$, its outcomes cannot be directly incorporated into the loss to guide optimisation. To address this, we introduce a vector $\tilde{A}_u$, representing what we refer to as \emph{promising actions} - that is, edge modifications that node $u$ should prioritise to escape its current community.

Therefore, we define the first term of the loss ($\ell_{\textit{hide}}$) as:

$$
    \ell_{\textit{hide}}(\hat{p}; A_u, \tilde{A}_u, q) = || \tilde{A}_u - (A_u + \hat{p})||_q,
$$

where $q \geq 1$.
In contrast, the second component of the loss ($\ell_{\textit{dist}}$) is designed to discourage large perturbations, aiming to identify the minimal counterfactual graph that causes $u$ to belong to a different community, according to $f$.
To this end, we assess the distance between the original and intermediate adjacency vectors:

$$
    \ell_{\textit{dist}}(\hat{p}; A_u, q) = || A_u - (A_u + \hat{p}) ||_q = || \hat{p} ||_q,
$$

where $q \geq 1$. 

We introduce the notion of a node's significance by assigning each node $v$ a real-valued score $S_v \in [0,1]$. Accordingly, each entry of $\tilde{A}_u$ is defined as:

- $\tilde{A}_{u,v} = (1-S_v)/2 \quad \text{if } v \in C_i,$
- $\tilde{A}_{u,v} = (1+S_v)/2 \quad \text{if } v \notin C_i.$    


If a node $v$ belongs to the same community as $u$ ($v \in C_i$) and has a high score ($S_v \approx 1$), then $\tilde{A}_{u,v} \approx 0$, which encourages the algorithm to disconnect from it if an edge exists. Conversely, if the node lies outside $C_i$ and also has a high score, $\tilde{A}_{u,v} \approx 1$, the algorithm is more likely to establish a connection if none exists. This reflects the aim of reducing cohesiveness within the community whilst strengthening connections outside of it. On the other hand, when a node has a low score ($S_v \approx 0$), $\tilde{A}_{u,v} \approx \frac{1}{2}$ in both scenarios, indicating no preference for adding or removing that connection, thus favoring no changes.

For each node, we calculate the values of $K$ structural properties, denoted by $\Omega=\{\omega_1\dots,\omega_K\}$. Then, we compute a $|V|$-dimensional ranking vector $r_{i}$ for each property $\omega_i \in \Omega$. Each element of this vector indicates the position of a node in the list of values for $\omega_i$, sorted in non-decreasing order. For example, consider a set of nodes $V = \{v_1,v_2,v_3\}$ and a property $\omega_i$, with values $[42,120,5]$, i.e., $\omega_i(v_1) = 42$, $\omega_i(v_2) = 120$, and $\omega_i(v_3) = 5$. Sorting these values in non-decreasing order yields $[5,42,120]$. The ranking vector $r_{i}$ assigns to each node the index (starting from 1) of its property value $\omega_i$ in the sorted list, resulting in $r_{i} = [2,3,1]$, i.e., $r_{i}[v_1] = 2$, as $\omega_i(v_1) = 42$ is the second element of the sorted list, and so on.
Thus, we normalise the rankings as follows:

$$
    S_v^{i} = \dfrac{r_{i}[v] - 1}{|V| - 1} \quad \forall v \in V, \forall i=1\dots K.
$$

The final scores are obtained by aggregating the individual scores associated with each property, for example through a linear combination: $S_v = \sum_{i=1}^K  a_i \, S_v^{i} \ \forall v \in V$, where $a_i \in [0,1]$ and $\sum_{i=1}^K a_i = 1$.

In this work, we consider the following structural properties of a node: $\Omega=${degree, betweenness centrality, intra/inter-community degree}. These properties are chosen to ensure consistency with the baselines methods, which also rely on them.
