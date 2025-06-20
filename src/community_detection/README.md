# src/community_detection

This folder contains the core implementations and interfaces for community detection algorithms and related functions.

## Contents

### Main files

- **algorithms.py**  
  Implements a unified class (`CommunityDetectionAlg`) that provides an interface for various community detection algorithms based on iGraph and external libraries. It supports classic algorithms (Greedy, Infomap, Louvain, Walktrap, etc.) and advanced algorithms such as SCD, DGCluster, and Leiden-LOCALE.

- **similarity_functions.py**  
  Contains the `CommunitySimilarity` class for computing similarity between communities, currently implementing the Sorensen similarity.

### Subfolder: extra_algs

This subfolder collects implementations of advanced or external algorithms:

- **scd.py**  
  Implementation of the SCD (Scalable Community Detection) algorithm, which optimizes clustering metrics for large graphs.

- **locale/**  
  Contains the implementation of the Leiden-LOCALE algorithm, which temporarily converts the graph to MatrixMarket format and calls an external function for community detection.

- **dgcluster/**  
  Implements DGCluster, a community detection algorithm based on Graph Neural Networks and clustering techniques.  
  - `dgcluster.py`: defines the model and clustering pipeline.
  - `dgcluster_training.py`: script for training and evaluating the DGCluster model.
  - `dgcluster_robustness.py`: analysis of the robustness of embeddings and DGCluster performance.
  - Subfolders `models/` and `results/`: contain pre-trained models and experiment results.

---

## Notes

- All algorithms return the detected communities as `NodeClustering` objects from the CDlib library.
- The structure is designed to be easily extendable with new algorithms or similarity metrics. 