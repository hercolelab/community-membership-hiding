from src.graph_environment.env import GraphEnvironment
from src.utils.cmh_experiment import CmhExperiment
from omegaconf import DictConfig
import hydra
import yaml
import logging

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
# Available evasion attack algorithms:
# - RAND (random),
# - DEG (degree),
# - BETW (betweenness),
# - ROAM (roam),
# - DICE (dice),
# - NABLA (nabla-cmh), 
# - DRL (drl-agent),   # not for now
# - GRE (greedy)       # not for now
#
# Suggested budget multiplier: [0.5,1,2]
# Suggested similarity threshold: [0.3,0.5,0.8]

# ------ EXPERIMENTS CONFIGURATION ------ #
graph_names = ["WORDS"]
#community_detection_algs = ["GRE", "LOUV", "WALK"]
community_detection_algs = ["WALK"]
#evasion_algs = ["RAND", "DEG", "BETW", "ROAM", "DICE", "NABLA"]  
evasion_algs = ["NABLA"]
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
        for community_detection_alg in community_detection_algs:
                # Set the environment 
                env = GraphEnvironment(
                    dataset, 
                    community_detection_alg,
                )
                # Create the CMH experiment
                cmh_experiment = CmhExperiment(
                    evasion_algs,
                    env,
                    verbose=False
                )
                for tau in taus:
                        for beta_factor in beta_factors:
                            # Set the parameters of the CMH problem
                            cmh_experiment.set_parameters(beta_factor, tau) 
                            # Print info
                            log.info("="*60)
                            log.info("COMMUNITY MEMBERSHIP HIDING") 
                            log.info("="*60)
                            log.info(f"Evading algorithms: {evasion_algs}")
                            log.info(f"Budget multiplier: {env.budget_multiplier}")
                            log.info(f"Budget: {env.budget}")
                            log.info(f"Similarity threshold: {env.tau}")
                            log.info("="*60)
                            log.info("START EXPERIMENT") 
                            log.info("="*60)
                            # Run the experiment
                            cmh_experiment.run_experiment()
                log.info("="*60)
                log.info("END EXPERIMENT") 
                log.info("="*60)
    log.info("="*60)
    log.info("END OF ALL EXPERIMENTS") 
    log.info("="*60)


if __name__ == "__main__":
    main()