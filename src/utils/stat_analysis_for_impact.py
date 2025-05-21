
from scipy.stats import ttest_ind
from scipy.stats import levene
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import json
from src.utils.utils import DatasetNames, EvasionAlgorithmsNames, DetectionAlgorithmsNames
from src.graph_environment.env import GraphEnvironment

def check_dir(path: str):
        """
        Check if the directory exists, if not create it.

        Parameters
        ----------
        path : str
            Path to the directory
        """
        if not os.path.exists(path):
            os.makedirs(path)


datasets = ["KAR", "WORDS", "VOTE", "POW", "FB_75", "COND_MAT"]
dataset_names = [getattr(DatasetNames, dataset).value for dataset in datasets]
evading_algs = ["RAND", "DEG", "BETW", "ROAM", "DICE", "DRL"] 

results_dir = "outputs_review/all_datasets/impact_significance"
save_dir = "outputs_review/all_datasets/impact_significance"

with open(f"{results_dir}/local_impact.json", "r") as f:
    local_impact = json.load(f)
                


for alg in evading_algs:
    statistics = {}
    for dataset in dataset_names:
        statistics[dataset] = {}
        x = local_impact[dataset]["NABLA"]
        y = local_impact[dataset][alg]
        
        # Levene's test
        levene_stat, p_levene = levene(x, y)
        # Check if the variances are equal
        if p_levene < 0.05:
            equal_var = False
        else:
            equal_var = True
        # Perform t-test
        t_stat, p_val = ttest_ind(x, y, equal_var=equal_var, alternative='less')
        
        # Store results
        statistics[dataset] = {
            "t_stat": t_stat,
            "p_val_one_tail": p_val,
            "p_val_levene": p_levene
        }
    
    # Save results into a JSON file
    check_dir(f"{save_dir}/{getattr(EvasionAlgorithmsNames, alg).value}")
    with open(f"{save_dir}/{getattr(EvasionAlgorithmsNames, alg).value}/t-test_tau_0.5_betaFactor_1.json", "w") as f:
        json.dump(statistics, f, indent=4)

                    
            
