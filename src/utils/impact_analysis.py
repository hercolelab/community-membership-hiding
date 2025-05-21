import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import json
from src.utils.utils import DatasetNames, EvasionAlgorithmsNames
from src.graph_environment.env import GraphEnvironment


datasets = ["KAR", "WORDS", "VOTE", "POW", "FB_75", "COND_MAT"]
dataset_names = [getattr(DatasetNames, dataset).value for dataset in datasets]
evading_algs = ["RAND", "DEG", "BETW", "ROAM", "DICE", "NABLA", "DRL"] 
seeds = [22,2025,42]

results_dir = "outputs_review/runs"

pageranks = {}
stats = {}
total_impact_dict = {}
local_impact_dict = {}

for dataset in datasets:
    env = GraphEnvironment(dataset, ["LEID"])
    g = env.original_graph.copy()
    pageranks[getattr(DatasetNames,dataset).value] = g.pagerank(directed=False)

    total_impact_dict[getattr(DatasetNames,dataset).value] = {}
    local_impact_dict[getattr(DatasetNames,dataset).value] = {}
    stats[getattr(DatasetNames,dataset).value] = {}
    
    for alg in evading_algs:
        total_impact_dict[getattr(DatasetNames,dataset).value][alg] = []
        local_impact_dict[getattr(DatasetNames,dataset).value][alg] = []
        stats[getattr(DatasetNames,dataset).value][alg] = {}
        for seed in seeds:
            json_path = f"{results_dir}/seed_{seed}/{getattr(DatasetNames,dataset).value}/leiden/tau_0.5/betaFactor_1/json_results/{getattr(EvasionAlgorithmsNames, alg).value}.json"
            with open(json_path, "r") as f:
                log = json.load(f)
            changes = log["changes"]
            total_impact = 0
            for change in changes:
                if "removed" in change:
                    removed = change["removed"]
                    added = change["added"]
                else:
                    removed = change["remove"]
                    added = change["add"]
                sum = 0
                count = 0
                local_impact = []
                for e in removed:
                    target_v = e[1]
                    associated_pagerank = pageranks[getattr(DatasetNames,dataset).value][target_v]
                    sum += associated_pagerank
                    count += 1
                for e in added:
                    target_v = e[1]
                    associated_pagerank = pageranks[getattr(DatasetNames,dataset).value][target_v]
                    sum += associated_pagerank
                    count += 1
                if count > 0:
                    local_impact = sum / count
                else:
                    local_impact = 0
                total_impact += local_impact
                local_impact_dict[getattr(DatasetNames,dataset).value][alg].append(local_impact)
            total_impact_dict[getattr(DatasetNames,dataset).value][alg].append(total_impact/len(changes))

        values = total_impact_dict[getattr(DatasetNames,dataset).value][alg]
        mean = np.mean(values)
        std = np.std(values)
        stats[getattr(DatasetNames,dataset).value][alg]["mean"] = mean
        stats[getattr(DatasetNames,dataset).value][alg]["std"] = std
        #stats[getattr(DatasetNames,dataset).value][alg] = mean

# Save the results to a JSON file
with open("outputs_review/all_datasets/impact_significance/total_impact.json", "w") as f:
    json.dump(stats, f, indent=4)
print("Results saved to outputs_review/all_datasets/impact_significance/total_impact.json")
# Save the results to a JSON file
with open("outputs_review/all_datasets/impact_significance/local_impact.json", "w") as f:
    json.dump(local_impact_dict, f, indent=4)
print("Results saved to outputs_review/all_datasets/impact_significance/local_impact.json")




