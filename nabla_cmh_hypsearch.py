from src.graph_environment.env import GraphEnvironment
from src.utils.cmh_experiment import CmhExperiment
from src.methods.nabla_cmh.config import HYPERPARAMETERS
from statistics import mean
import hydra
import logging
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import wandb
import os
import yaml
import json
import numpy as np

log = logging.getLogger(__name__)


def exp(env: GraphEnvironment, save_path: str, wandb_cfg = None):
    """"
    This function runs the experiment with the given configuration and saves the results.

    Parameters
    ----------
    env : GraphEnvironment
        The environment to use for the experiment.
    save_path : str
        The path to save the results.
    wandb_cfg : dict
        The configuration for wandb.
    """

    hyps = HYPERPARAMETERS

    with wandb.init(config=wandb_cfg):
        wandb_cfg = wandb.config
        run_name = wandb.run.name
        
        log.info("Run name: {} - Dataset: {} - Detection Algorithm: {}".format(run_name,env.graph_name_output, env.community_detection_alg_name_output))
        log.info(f"Output directory: {save_path}")

        # Problem Configuration
        dataset_name = env.graph_name_output
        train_alg = "greedy"
        test_alg = env.community_detection_alg_name_output
        tau = env.tau
        c_beta = env.budget_multiplier

        #Candidate hyperparameters
        evader_hyps = hyps[dataset_name][f"training_{train_alg}"][f"testing_{test_alg}"]
        evader_hyps[f"tau_{tau}"][f"betaFactor_{c_beta}"]["T"] = wandb_cfg.max_it
        evader_hyps[f"tau_{tau}"][f"betaFactor_{c_beta}"]["lr"] = wandb_cfg.lr
        evader_hyps[f"tau_{tau}"][f"betaFactor_{c_beta}"]["lambd"] = wandb_cfg.lambd
        dirichlet_seed = wandb_cfg.dirichlet_seed
        np.random.seed(int(dirichlet_seed*1e6))
        coeffs = np.random.dirichlet([1, 1, 1, 1]).tolist()
        evader_hyps[f"tau_{tau}"][f"betaFactor_{c_beta}"]["promising_action_coeffs"] = coeffs
        log.info(f"Evader configuration: {evader_hyps[f'tau_{tau}'][f'betaFactor_{c_beta}']}")

        # Initialize the test class
        experiment = CmhExperiment(evasion_algs=["NABLA"], env=env, verbose=False)
        experiment.run_experiment()

        # Open the JSON file for wandb log
        evaluation_path = save_path + f"/{dataset_name}/{test_alg}/tau_{tau}/betaFactor_{c_beta}/json_results/nabla-cmh.json"
        with open(evaluation_path, 'r') as eval_file:
            results = json.load(eval_file)

        #Compute the F1 score
        f1_score = mean([0 if (results["goal"][i] + results["nmi"][i]) == 0 
                 else 2 * (results["goal"][i] * results["nmi"][i]) / (results["goal"][i] + results["nmi"][i])
                 for i in range(len(results["goal"]))])
        time = mean([results["time"][i] for i in range(len(results["time"]))])
        log.info(f"F1 score: {f1_score}")
        log.info(f"Time: {time}")
        wandb.log({"f1": f1_score})
        wandb.log({"time": time})




# ------ EVASION OPTIONS ------ #
# For this script is allowed to use just a single dataset, budget and target node.
# This script allows to run multiple evasion attack algorithms
#
# Available datasets: 
# - KAR (0-33 nodes), 
# - WORDS (0-111 nodes), 
# - VOTE (0-888 nodes), 
# - POW (0-4940 nodes), 
# - FB_75 (0-6385 nodes), 
# - COND_MAT (0-23132 nodes)
#
# Available community detection algorithms: 
# - GRE (greedy), 
# - INF (infomap), 
# - LAB (label propagation), 
# - LOUV (louvain), 
# - WALK (walktrap)
#
# Suggested budget multiplier: [0.5,1,2]
# Suggested similarity threshold: [0.3,0.5,0.8]


@hydra.main(config_path="src/conf", config_name="hyp_search", version_base=None)
def main(cfg: DictConfig) -> None:
    
    save_path = HydraConfig.get().runtime.output_dir

    # ------ EXPERIMENTS CONFIGURATION ------ #
    graph_name = "POW"
    alg = "WALK"
    tau = 0.5
    c_beta = 0.5

    # ------ UPDATE HYDRA CONFIG FILE ------ #
    with open("src/conf/hyp_search.yaml", "r") as file:
            cfg = yaml.safe_load(file)
    cfg["graphs"] = graph_name
    cfg["community_detection_algs"] = alg
    cfg["evasion_attack_algs"] = "NABLA"
    cfg["budget_multipliers"] = tau
    cfg["similarity_thresholds"] = c_beta
    with open("src/conf/hyp_search.yaml", "w") as file:
        yaml.dump(cfg, file, sort_keys=False)

    # ---- Environment Setup ---- #
    env = GraphEnvironment(
        graph_name=graph_name,
        community_detection_alg=alg,
        budget_multiplier=c_beta,
        similarity_threshold=tau,
    )

    # ---- Search Configuration ---- #

    sweep_config = {
        "method": "random",
        "metric": {"name": "f1", "goal": "maximize"},
        "parameters": {
            "max_it": {"values": list(range(50, 160,10))},
            "lr": {"min": 0.0001, "max": 0.01},
            "lambd": {"min": 1.0, "max": 25.0},
            "dirichlet_seed": {"min": 0.0, "max": 1.0},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=f"nabla-cmh_search {graph_name} {alg} tau_{tau} beta_{c_beta}")
    wandb.agent(sweep_id, function=lambda: exp(env,save_path), count=1000)


if __name__ == "__main__":
    main()

