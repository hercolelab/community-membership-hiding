# Community Membership Hiding

Community detection is a fundamental problem in network science, where algorithms attempt to uncover groups of nodes, called *communities*, that share strong internal connections. While this task is widely applied in areas such as social networks, recommendation systems, and biological data analysis, it raises significant privacy concerns. Being identified as a member of a community can expose sensitive affiliations, such as political, religious, or professional memberships, which individuals may wish to keep private.

The **Community Membership Hiding (CMH)** problem addresses this issue by strategically modifying a network’s structure to obscure a target node’s membership in a specific community. Given a network and a community detection algorithm, the goal is to perturb the network in a way that prevents the target node from being recognized as part of its original community.

This repository is built on *igraph*, and provides an implementation of **∇-CMH**, a gradient-based optimisation approach, a Deep Reinforcement Learning (**DRL-agent**) method, and several baselines, such as **DICE**, **ROAM**, Random-based, Degree-based, and Centrality-based. 
It supports multiple real-world network datasets and different community detection algorithms.

## Problem Definition

Let $G = (V, E)$ be an undirected graph where $v$ is the set of nodes and $E$ is the set of edges. A **community detection algorithm** $f(\cdot)$ partitions the nodes into non-overlapping communities $\{C_1, C_2, ..., C_k\}$, where each node belongs to exactly one community. Given a **target node** $u$ that belongs to a community $C_i$, the objective of the CMH problem is to modify the structure of $G$ such that, when the community detection algorithm $f(\cdot)$ is applied to the modified graph $G'$, the node $u$ is no longer recognized as a member of $C_i$.

<p align="center">
  <img src="images/cmh.gif" width="50%" />
</p>


To achieve this, we define a **perturbation function** $h_{\theta}(\cdot)$, parameterized by $\theta$, that modifies $g$ into a new graph $G' = h_{\theta}(G)$ by adding or removing edges in the neighborhood of $u$. The goal is to find an optimal function $h^*_{\theta}$ that ensures $u$ is removed from its original community:

$$\theta^* = \arg\min_{\theta} L(h_{\theta}; G, f, u)$$

subject to a **budget constraint** on the number of modifications $|B| \leq \beta$, where $B$ is the set of edges modified, and $\beta$ represents the maximum number of allowed changes.

The effectiveness of the hiding process is measured using a **similarity function** $sim(C_i \setminus \{u\}, C'_i \setminus \{u\})$, which compares the original community $C_i$ and the new assigned community $C'_i$ in the modified graph $G'$. 
The hiding task is considered successful if:

$$
\text{sim}(C_i \setminus \{u\}, C'_i \setminus \{u\}) \leq \tau
$$

where $\tau$ is a predefined threshold controlling the required level of dissimilarity, which ranges between $0$ and $1$.


## Installation

Make sure to have `conda` installed:

<pre> <code> conda create --name graph-cmh python=3.9.18 
  conda activate graph-cmh </code> </pre>

Then, install the requirements:

<pre> <code> pip install -r requirements.txt </code> </pre>

If you have a GPU, run (according to [CUDA version](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)): 

<pre> <code> pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html </code> </pre>

If you want to run on CPU: 

<pre> <code> pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cpu.html </code> </pre>

## Usage

1. **Run the CMH-experiment**:
<pre> <code> python main.py </code> </pre>
where the hyperparameters of the experiment, i.e. 
- *dataset* : `[KAR, WORDS, VOTE, POW, FB_75, COND_MAT]`
- *detection algorithm* : `[GRE, LOUV, LEID, INF, LAB, WALK, SCD, LOC, DGC]`
- *hiding method* : `[NABLA, DRL, DICE, ROAM, RAND, DEG, BETW]`
- *budget factor* $\beta$: `[0.5,1,2]`
- *similarity threshold* $\tau$: `[0.3,0.5,0.8]`
  
can be modified at lines 51-64 of `main.py`.   
To change the seed of the experiment, refer to line 92 of `src/utils/utils.py`.

2. **Run a single evasion**
<pre> <code> python single_evasion.py </code> </pre>
where the hyperparameters of the experiment can be modified at lines 61-68 of `single_evasion.py`.   
This script does not support multiple datasets, betas and taus.

## Directory Structure

```bash
├── dataset
│   ├── networks                            # Contains the graphs
│   │   └── ...
│   └── README.md
├── images                                  
│   └── ...
├── notebooks                               # Contains the notebooks used for the analysis
│   └── ...
├── outputs                                 # Contains the results of new runs
│   └── ...                                                                                                    
├── outputs_review                          # Contains the final results of the study
│   ├── all_datasets                        
│   │   └── ...
│   ├── dataset_analysis
│   │   └── ...
│   └── runs
│       └── ...
├── src                                     # Contains the source code
│   ├── baselines
│   │   ├── betweenneess.py
│   │   ├── degree.py
│   │   ├── dice.py
│   │   ├── random.py
│   │   ├── roam.py
│   │   └── README.md
│   ├── community_detection                 # Contains community detection algorithms
│   │   ├── extra_algs
│   │   │   ├── dgcluster
│   │   │   │   ├── models
│   │   │   │   ├── results
│   │   │   │   ├── dgcluster_robustness.py
│   │   │   │   ├── dgcluster_training.py
│   │   │   │   ├── dgcluster.py
│   │   │   │   └── README.md
│   │   │   ├── locale                 
│   │   │   │   ├── temp_graphs
│   │   │   │   │   └── ...
│   │   │   │   ├── locale.py
│   │   │   │   └── README.md.py
│   │   │   ├── scd.py
│   │   │   └── README.md
│   │   ├── algorithms.py                  
│   │   ├── similarity_functions.py
│   │   └── README.md
│   ├── conf                               # Cointains the yaml files for Hydra logs
│   │   └── ...
│   ├── graph_environment
│   │   ├── env.py              
│   │   └── README.md
│   ├── methods                             
│   │   ├── drl_agent                      # Cointains the Deep Reinforcement Learning agent     
│   │   │   ├── a2c
│   │   │   │   └── ...
│   │   │   ├── models
│   │   │   │   └── ...
│   │   │   ├── agent.py
│   │   │   └── README.md
│   │   └── nabla_cmh                      # Cointains the Nabla-CMH method    
│   │       ├── config.py
│   │       ├── nabla_cmh-hyp_search.py
│   │       ├── nabla_cmh.py
│   │       ├── nabla_utils.py
│   │       └── README.md
│   └── utils                              # Cointains the utils functions
│       ├── cmh_experiment.py          
│       ├── utils.py
│       └── README.md
│
├── dataset_analysis.py
├── main.py
├── single_evasion.py
├── README.md
└── requirements.txt
```


