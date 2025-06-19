# outputs_review

This folder collects all results, analyses, and statistics generated during experiments and evaluations of datasets and community detection algorithms.

## Folder Structure

- **all_datasets/**  
  Contains aggregated results and comparative analyses across all datasets and algorithms.
  - `f1_score/`: Performance evaluations (F1-score) for each algorithm and dataset, organized by parameters such as `tau`.
  - `impact_significance/`: Statistical analyses on the impact of actions and metrics (e.g., betweenness, degree, dice), including significance tests (t-test) and visualizations (e.g., `dot-plot.png`).
  - Other files (PDFs, images): Summary tables, plots, and reports.

- **dataset_analysis/**  
  Contains `.json` files with descriptive statistics and specific analyses for each dataset (e.g., `cond-mat.json`, `dblp.json`, etc.).

- **runs/**  
  Detailed results of experiment runs, organized by random seed (`seed_22`, `seed_42`, etc.), dataset, and algorithm.
  - Example structure:
    ```
    runs/seed_22/cond-mat/dgcluster/
    ```
    Each folder contains the results for the corresponding run.

## Usage

This folder is intended to:
- Store the results of analyses and experiments.
- Facilitate comparison between different algorithms and datasets.
- Provide data ready for generating tables, plots, and reports.

## Notes

- `.json` files contain structured data that can be easily loaded in Python or other languages for further analysis.
- Images and PDFs are useful for documentation and presentation of results. 