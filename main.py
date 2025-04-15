from src.graph_environment.env import GraphEnvironment
from src.utils.cmh_experiment import CmhExperiment
from src.utils.utils import Utils
from omegaconf import DictConfig
import hydra
import yaml
import logging
from rich.logging import RichHandler

"""
# ------ EVASION OPTIONS ------ #
This script supports running multiple evasion attack algorithms on several datasets, 
and detection algorithms, with a specified budget and similarity threshold.

Available Datasets:
    - KAR:       0-33 nodes
    - WORDS:     0-111 nodes
    - VOTE:      0-888 nodes
    - POW:       0-4940 nodes
    - FB_75:     0-6385 nodes
    - COND_MAT:  0-23132 nodes

Available Community Detection Algorithms:
    - GRE:  Greedy
    - INF:  Infomap
    - LAB:  Label Propagation
    - LOUV: Louvain
    - WALK: Walktrap

Available Evasion Attack Algorithms:
    - RAND:  Random
    - DEG:   Degree
    - BETW:  Betweenness
    - ROAM:  Roam
    - DICE:  Dice
    - NABLA: Nabla-CMH
    - DRL:   DRL-Agent (not supported yet)
    - GRE:   Greedy (not supported yet)

Suggested Parameters:
- Budget Multipliers (beta factor): [0.5, 1, 2]
- Similarity Thresholds (tau): [0.3, 0.5, 0.8]
"""

# ------ EXPERIMENTS CONFIGURATION ------ #
graph_names = ["KAR","WORDS"]
community_detection_algs = ["GRE", "LOUV", "WALK"]
#community_detection_algs = ["WALK"]
evasion_algs = ["RAND", "DEG", "BETW", "ROAM", "DICE", "NABLA"] 
#evasion_algs = ["RAND", "DEG", "BETW", "ROAM", "DICE"]  
#evasion_algs = ["NABLA"]
beta_factors = [0.5,1,2]
#beta_factors = [1]
taus = [0.5]  

# ------ UPDATE HYDRA CONFIG FILE ------ #
with open("src/conf/experiment.yaml", "r") as file:
        cfg = yaml.safe_load(file)
cfg["graphs"] = graph_names
cfg["community_detection_algs"] = community_detection_algs
cfg["evasion_attack_algs"] = evasion_algs
cfg["budget_multipliers"] = beta_factors
cfg["similarity_thresholds"] = taus
with open("src/conf/experiment.yaml", "w") as file:
    yaml.dump(cfg, file, sort_keys=False)


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