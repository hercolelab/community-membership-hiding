# Project At A Glance: Community Membership Hiding

Community Membership Hiding (CMH) studies how to modify a graph so that a target node is no longer detected as part of its original community. This repository implements CMH experiments on top of `igraph`, including a gradient-based Nabla-CMH method, a Deep Reinforcement Learning agent, and baseline strategies such as DICE, ROAM, random, degree-based, and betweenness-based hiding. It supports multiple graph datasets, several community detection algorithms, and stores both reproducible experiment outputs and final analysis artifacts.

## Repository Structure Analysis

```text
community-membership-hiding/
├── AGENTS.md                         # Agent-facing project notes and repository map.
├── README.md                         # Main project overview, installation steps, usage, and high-level directory guide.
├── requirements.txt                  # Python dependencies for running experiments and analyses.
├── main.py                           # Main multi-configuration CMH experiment entry point.
├── single_evasion.py                 # Single-run evasion script for one dataset/algorithm/method setup.
├── dataset/                          # Input graph datasets and dataset documentation.
│   ├── README.md                     # Dataset descriptions and loading notes.
│   └── networks/                     # Raw graph files in txt/mtx formats.
│       └── warpcast/                 # Warpcast graph snapshots at different sizes.
├── images/                           # README and documentation images/animations.
├── notebooks/                        # Analysis and result-reconstruction notebooks.
├── outputs_review/                   # Curated/final experiment outputs used for study review.
│   ├── README.md                     # Notes for reviewed output artifacts.
│   ├── all_datasets/                 # Cross-dataset summaries, plots, and statistical results.
│   ├── dataset_analysis/             # Per-dataset graph statistics in JSON form.
│   ├── feature_analysis/             # Feature correlation tables, timing data, and heatmaps.
│   └── runs/                         # Full reviewed runs grouped by seed, dataset, algorithm, tau, and budget.
└── src/                              # Core implementation package.
    ├── README.md                     # Source-level overview.
    ├── baselines/                    # Non-learning and heuristic hiding baselines.
    │   ├── README.md                 # Baseline method notes.
    │   ├── betweenness.py            # Betweenness-centrality based edge modification strategy.
    │   ├── degree.py                 # Degree-based edge modification strategy.
    │   ├── dice.py                   # DICE baseline implementation.
    │   ├── random.py                 # Random edge perturbation baseline.
    │   └── roam.py                   # ROAM baseline implementation.
    ├── community_detection/          # Community detection wrappers and similarity logic.
    │   ├── README.md                 # Community detection module notes.
    │   ├── algorithms.py             # Main interface for supported detection algorithms.
    │   ├── similarity_functions.py   # Community similarity metrics used to evaluate hiding.
    │   └── extra_algs/               # Additional/custom community detection implementations.
    │       ├── scd.py                # SCD algorithm support.
    │       ├── locale/               # Locale community detection implementation.
    │       └── dgcluster/            # DGCluster implementation, training scripts, models, and results.
    ├── conf/                         # Hydra YAML configs for experiments and analyses.
    ├── graph_environment/            # Graph environment used by learning-based methods.
    │   ├── README.md                 # Environment notes.
    │   └── env.py                    # Environment dynamics for graph modification actions.
    ├── methods/                      # Main CMH methods.
    │   ├── README.md                 # Method-level overview.
    │   ├── drl_agent/                # Deep Reinforcement Learning hiding agent.
    │   │   ├── README.md             # DRL agent notes.
    │   │   ├── agent.py              # Agent orchestration.
    │   │   ├── a2c/                  # Actor-Critic implementation components.
    │   │   └── models/               # Saved DRL model checkpoints.
    │   ├── nabla_cmh/                # Main Nabla-CMH gradient-based method.
    │   ├── nabla_cmh_projection/     # Projected variant of Nabla-CMH.
    │   └── nabla_cmh_scoreNN/        # Score neural-network variant/prototype for Nabla-CMH.
    └── utils/                        # Experiment orchestration, analysis, plotting, and shared helpers.
        ├── cmh_experiment.py         # CMH experiment runner utilities.
        ├── dataset_analysis.py       # Dataset statistics/analysis utilities.
        ├── feature_analysis.py       # Feature analysis routines.
        ├── impact_analysis.py        # Impact analysis routines.
        ├── stat_analysis_for_impact.py # Statistical analysis for impact results.
        └── utils.py                  # Shared helper functions and global utilities.
```
