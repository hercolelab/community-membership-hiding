from src.graph_environment.env import GraphEnvironment
from src.utils.cmh_experiment import CmhExperiment
from src.utils.utils import Utils, ExperimentHyps
from omegaconf import DictConfig
import hydra
import yaml
import logging
import json

"""
# ------ EVASION OPTIONS ------ #
This script supports running multiple evasion attack algorithms on several datasets, 
and detection algorithms, with a specified budget and similarity threshold.

Available Datasets:
    - KAR: Zachary Karate Club    
    - WORDS: David Copperfield Words    
    - VOTE: Wikipedia Voting    
    - POW: U.S. Power Grid   
    - FB_75: Facebook Friendships
    - COND_MAT: Condense Matter Collaborations

Available Community Detection Algorithms:
    - GRE:  Greedy
    - INF:  Infomap
    - LAB:  Label Propagation
    - LOUV: Louvain
    - WALK: Walktrap
    - LEID: Leading Eigenvector
    - SCD:  Scalable Community Detection
    - LEID: Leiden
    - LOC: Locale
    - DGC: DGCluster

Available Evasion Attack Algorithms:
    - RAND:  Random
    - DEG:   Degree
    - BETW:  Betweenness
    - ROAM:  Roam
    - DICE:  Dice
    - NABLA: Nabla-CMH
    - DRL:   DRL-Agent     
    - GRE:   Greedy (not supported yet)

Suggested Parameters:
- Budget Multipliers (beta factor): [0.5, 1, 2]
- Similarity Thresholds (tau): [0.3, 0.5, 0.8]
"""

# ------ EXPERIMENTS CONFIGURATION ------ #
graph_names = ["KAR", "WORDS", "VOTE", "POW", "FB_75", "COND_MAT"]
#graph_names = ["KAR"]

#community_detection_algs = ["GRE", "LOUV", "LEID", "WALK", "INF", "LAB", "SCD", "LOC", "DGC"]
community_detection_algs = ["DGC"]



evasion_algs = ["RAND", "DEG", "BETW", "ROAM", "DICE", "NABLA", "DRL"] 
#evasion_algs = ["NABLA", "DICE"]

beta_factors = [0.5,1,2]
#beta_factors = [1]

#taus = [0.3, 0.5, 0.8]
taus = [0.5]  

# ------ UPDATE HYDRA CONFIG FILE ------ #
with open("src/conf/experiment.yaml", "r") as file:
        cfg = yaml.safe_load(file)
cfg["graphs"] = graph_names
cfg["community_detection_algs"] = community_detection_algs
cfg["evasion_attack_algs"] = evasion_algs
cfg["budget_multipliers"] = beta_factors
cfg["similarity_thresholds"] = taus
cfg["seed"] = ExperimentHyps.seed.value
with open("src/conf/experiment.yaml", "w") as file:
    yaml.dump(cfg, file, sort_keys=False)


# ------ MAIN FUNCTION ------ #
log = logging.getLogger(__name__)
@hydra.main(config_path="src/conf", config_name="experiment", version_base=None)
def main(cfg:DictConfig) -> None:
    """
    Main function to run the Community Membership Hiding experiment.
    """
    
    
    for dataset in graph_names:
        # Set the environment 
        env = GraphEnvironment(
            dataset, 
            community_detection_algs,
        )
        # Create the CMH experiment
        cmh_experiment = CmhExperiment(
            evasion_algs,
            env,
        )
        for tau in taus:
                for beta_factor in beta_factors:
                    # Set the parameters of the CMH problem
                    cmh_experiment.set_parameters(beta_factor, tau) 
                    # Print info
                    log.info(" ")
                    log.info("========== COMMUNITY MEMBERSHIP HIDING ==========")
                    log.info(" ")
                    log.info("Experiment configuration:")
                    log.info("  • Evasion algorithms   : %s", evasion_algs)
                    log.info("  • Budget multiplier    : %.2f", env.budget_multiplier)
                    log.info("  • Budget               : %d", env.budget)
                    log.info("  • Similarity threshold : %.2f", env.tau)
                    log.info(">> Starting experiment...")
                    # Run the experiment
                    cmh_experiment.run_experiment()
                    log.info(">> Experiment completed.")

    log.info(">> Computing metrics and plots...") 
    Utils.plot_metrics(
        datasets=graph_names,
        evasion_algs=evasion_algs,
        detection_algs=community_detection_algs,
        budget_factors=beta_factors,
        taus=taus,
        metrics = ["goal","nmi","f1","time","steps"],
    )
    log.info("========== END OF ALL EXPERIMENTS ==========")


if __name__ == "__main__":
    main()