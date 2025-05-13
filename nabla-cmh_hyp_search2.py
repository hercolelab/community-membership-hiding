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
from scipy.special import softmax

log = logging.getLogger(__name__)


# ------ EXPERIMENT FUNCTION ------ #
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

        # Problem Configuration
        dataset_name = env.graph_name_output
        train_alg = "greedy"
        tau = env.tau
        c_beta = env.budget_multiplier

        #Candidate hyperparameters
        evader_hyps = hyps[dataset_name][f"training_{train_alg}"]
        evader_hyps[f"tau_{tau}"][f"betaFactor_{c_beta}"]["T"] = wandb_cfg.max_it
        evader_hyps[f"tau_{tau}"][f"betaFactor_{c_beta}"]["lr"] = wandb_cfg.lr
        raw_weights = np.array([wandb_cfg.p1, wandb_cfg.p2, wandb_cfg.p3, wandb_cfg.p4])
        coeffs = softmax(raw_weights)
        evader_hyps[f"tau_{tau}"][f"betaFactor_{c_beta}"]["promising_action_coeffs"] = coeffs
        log.info(f"Evader configuration: {evader_hyps[f'tau_{tau}'][f'betaFactor_{c_beta}']}")


        # Initialize the test class
        experiment = CmhExperiment(evasion_algs=["NABLA"], env=env)
        experiment.set_parameters(c_beta, tau) 
        experiment.run_experiment()

        # Open the JSON file for wandb log
        evaluation_path = save_path + f"/{dataset_name}/{train_alg}/tau_{tau}/betaFactor_{c_beta}/json_results/nabla-cmh.json"
        with open(evaluation_path, 'r') as eval_file:
            results = json.load(eval_file)

        #Compute the F1 score
        goal_mean = mean(results["goal"])
        nmi_mean = mean(results["nmi"])
        if goal_mean + nmi_mean == 0:
            f1_score = 0
        else:
            f1_score = 2 * goal_mean * nmi_mean / (goal_mean + nmi_mean)
        time = mean([results["time"][i] for i in range(len(results["time"]))])
        steps = (
            mean([results["steps"][i] for i in range(len(results["goal"])) if results["goal"][i] == 1]) / env.budget
            if any(results["goal"][i] == 1 for i in range(len(results["goal"])))
            else 0
        )
        log.info(f"F1 score: {f1_score}")
        log.info(f"Steps: {steps}")
        log.info(f"Time: {time}")
        wandb.log({"f1": f1_score})
        wandb.log({"steps": steps})
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
    graph_name = "KAR"
    alg = ["GRE"]
    tau = 0.5
    c_beta = 1

    # ------ UPDATE HYDRA CONFIG FILE ------ #
    with open("src/conf/hyp_search.yaml", "r") as file:
            cfg = yaml.safe_load(file)
    cfg["graph"] = graph_name
    cfg["community_detection_alg"] = alg
    cfg["evasion_attack_alg"] = "NABLA"
    cfg["budget_multiplier"] = c_beta
    cfg["similarity_thresholds"] = tau
    with open("src/conf/hyp_search.yaml", "w") as file:
        yaml.dump(cfg, file, sort_keys=False)

    # ---- Environment Setup ---- #
    env = GraphEnvironment(
        graph_name=graph_name,
        community_detection_algs=alg,
        budget_multiplier=c_beta,
        similarity_threshold=tau,
    )

    # ---- Search Configuration ---- #


    sweep_config = {
        "method": "bayes",  
        "metric": {
            "name": "f1",  
            "goal": "maximize"
        },
        "parameters": {
            # --- GD hyperparameters --- #
            "max_it": {
                "distribution": "q_uniform",  
                "min": 300,
                "max": 500,
                "q": 10
            },
            "lr": {
                "distribution": "log_uniform",  
                "min": np.log(0.02),
                "max": np.log(0.1)
            },
            
            # --- Promising actions weights --- #
            "p1": {"distribution": "normal", "mu": 0, "sigma": 1},  
            "p2": {"distribution": "normal", "mu": 0, "sigma": 1},
            "p3": {"distribution": "normal", "mu": 0, "sigma": 1},
            "p4": {"distribution": "normal", "mu": 0, "sigma": 1},
        }
    }


    sweep_id = wandb.sweep(sweep_config, project=f"hyp_search2.0 {graph_name} {alg[0]} tau_{tau} beta_{c_beta}")
    wandb.agent(sweep_id, function=lambda: exp(env,save_path), count=150)


if __name__ == "__main__":
    main()

