from src.graph_environment.env import GraphEnvironment
import logging
import hydra
import yaml
from omegaconf import DictConfig
from src.baselines.random import RandomHiding

# ------ Evasion details ------ #
# For this script is allowed to use just a single dataset, community detection algorithm and budget
graph_name = "KAR"
community_detection_alg = "LOUV"
target_node = 11
budget_multiplier = 1

# ------ Update the configuration file ------ #
with open("src/conf/single_evasion.yaml", "r") as file:
        cfg = yaml.safe_load(file)
cfg["graph"] = graph_name
cfg["community_detection"] = community_detection_alg
cfg["target_node"] = target_node
cfg["budget_multiplier"] = budget_multiplier
with open("src/conf/single_evasion.yaml", "w") as file:
    yaml.dump(cfg, file, sort_keys=False)


# ------ Main function ------ #
log = logging.getLogger(__name__)
@hydra.main(config_path="src/conf", config_name="single_evasion", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Starting the main function")

    # Set the environment 
    env = GraphEnvironment(graph_name, community_detection_alg)
    # Get the budget
    mu = env.get_average_budget()
    budget = mu * budget_multiplier
    # Set the evasion algorithms
    evasion_alg = RandomHiding(env, target_node, budget)
    # Results
    new_graph, steps, changes = evasion_alg.hide_target_node_from_community()
    # Compute metrics
    ## TODO 

    # Print results
    log.info(f"Target node: {target_node}")
    log.info(f"Number of steps: {steps}")
    log.info(f"Changes: {changes}")



    log.info("Finishing the main function")

if __name__ == "__main__":
    main()
