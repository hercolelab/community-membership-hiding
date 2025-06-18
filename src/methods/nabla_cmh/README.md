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