from src.graph_environment.env import GraphEnvironment
from src.utils.utils import EvasionAlgorithmsNames, DRL_agentHyps, FilePaths
import logging
import hydra
import yaml
from src.baselines.random import RandomHiding
from src.baselines.degree import DegreeHiding
from src.baselines.betweenness import CentralityHiding
from src.baselines.roam import RoamHiding
from src.baselines.dice import DiceHiding
from src.methods.nabla_cmh.nabla_cmh import nablaCMH
from src.methods.drl_agent.agent import Agent
from time import time
import json
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


"""
# ------ EVASION OPTIONS ------ #
This script supports running multiple evasion attack algorithms on a single dataset, 
with a specified budget and target node.

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
    - LEID: Leading Eigenvector
    - SCD:  Scalable Community Detection
    - LEID: Leiden
    - LOC: Locale

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


# ------ ATTACK CONFIGURATION ------ #
graph_name = "KAR"
#community_detection_alg = ["GRE", "LOUV", "WALK", "INF", "LAB", "EIG", "SCD",  "LOC"]
community_detection_alg = ["LOC"]
evasion_attack_algs = ["RAND", "DEG", "BETW", "ROAM", "DICE", "NABLA", "DRL"]
#evasion_attack_algs = ["DRL"]
target_node = 22
budget_multiplier = 2
similarity_threshold = 0.5

# ------ UPDATE HYDRA CONFIG FILE ------ #
with open("src/conf/single_evasion.yaml", "r") as file:
        cfg = yaml.safe_load(file)
cfg["graph"] = graph_name
cfg["community_detection"] = community_detection_alg
cfg["evasion_attack_algs"] = evasion_attack_algs
cfg["target_node"] = target_node
cfg["budget_multiplier"] = budget_multiplier
cfg["similarity_threshold"] = similarity_threshold
with open("src/conf/single_evasion.yaml", "w") as file:
    yaml.dump(cfg, file, sort_keys=False)


# ------ MAIN FUNCTION ------ #
log = logging.getLogger(__name__)
@hydra.main(config_path="src/conf", config_name="single_evasion", version_base=None)
def main(cfg: DictConfig) -> None:

    # Set the environment 
    env = GraphEnvironment(
        graph_name, 
        community_detection_alg,
        target_node=target_node,
        budget_multiplier=budget_multiplier,
        similarity_threshold=similarity_threshold
    )

    # Results dictionary
    results = {
        "graph": env.graph_name_output,
        "community_detection": env.community_detection_alg_names_output,
        "target_node": target_node,
        "community_size": env.target_community_size,
        "budget_multiplier": budget_multiplier,
        "similarity_threshold": similarity_threshold,
        "budget": env.budget,
    }
    
    # Print info
    log.info("="*60)
    log.info("SINGLE COMMUNITY MEMBERSHIP HIDING") 
    log.info("="*60)
    log.info(f"Target node: {target_node}")
    log.info(f"Community size: {env.target_community_size}")
    log.info(f"Budget: {env.budget}")
    log.info(f"Similarity threshold: {similarity_threshold}")

    for alg in evasion_attack_algs:
        # Set the evasion attack algorithm
        if alg == EvasionAlgorithmsNames.RAND.name:
            evasion_alg = RandomHiding(env, target_node, env.budget)
        elif alg == EvasionAlgorithmsNames.DEG.name:
            evasion_alg = DegreeHiding(env, target_node, env.budget)
        elif alg == EvasionAlgorithmsNames.BETW.name:
            evasion_alg = CentralityHiding(env, target_node, env.budget)
        elif alg == EvasionAlgorithmsNames.ROAM.name:
            evasion_alg = RoamHiding(env, target_node, env.budget)
        elif alg == EvasionAlgorithmsNames.DICE.name:
            evasion_alg = DiceHiding(env, target_node, env.budget)
        elif alg == EvasionAlgorithmsNames.NABLA.name:
            evasion_alg = nablaCMH(env, target_node, env.budget)
        elif alg == EvasionAlgorithmsNames.DRL.name:
            evasion_alg = Agent(env)
        else:
            raise ValueError("Invalid evasion attack algorithm")
        
        # Set the hiding function
        if alg == EvasionAlgorithmsNames.NABLA.name:
            func_call = lambda: evasion_alg.community_membership_hiding(verbose_iterations=True)  
        elif alg == EvasionAlgorithmsNames.DRL.name:
                            func_call = lambda: evasion_alg.test(
                                lr=DRL_agentHyps.LR_EVAL.value,
                                gamma=DRL_agentHyps.GAMMA_EVAL.value,
                                lambda_metric=DRL_agentHyps.LAMBDA_EVAL.value,
                                alpha_metric=DRL_agentHyps.ALPHA_EVAL.value,
                                epsilon_prob=DRL_agentHyps.EPSILON_EVAL.value,
                                model_path=FilePaths.TRAINED_MODEL.value,
                            )  
        else:
            func_call = lambda: evasion_alg.community_membership_hiding()  

        # Counterfactual graph
        start = time()
        result = func_call()
        end = time()
        total_time = end - start

        # Unpack variables based on algorithm type
        if alg == EvasionAlgorithmsNames.NABLA.name:
            new_graph, steps, changes, add_results = result  # nabla-cmh returns an extra value
        else:
            new_graph, steps, changes = result  

        # Compute metrics
        goal, nmi = env.get_metrics(new_graph)

        # Print and save results
        results[getattr(EvasionAlgorithmsNames,alg).value]= {
            "steps": steps,
            "changes": changes,
            "goal": goal,
            "nmi": nmi,
            "time": total_time
        }
        if alg == EvasionAlgorithmsNames.NABLA.name:
            results[getattr(EvasionAlgorithmsNames,alg).value]["additional_results"] = add_results
            
        log.info("="*60)
        log.info(f"Results for {getattr(EvasionAlgorithmsNames,alg).value} evasion attack")
        log.info("="*60)
        log.info(f"Steps: {steps}")
        #log.info(f"Changes: {changes}")
        log.info(f"Goal: {goal}")
        log.info(f"NMI: {nmi}")
        log.info(f"Time: {total_time}")
        
    log.info("="*60)

    # Save results in a json file
    dir_path = HydraConfig.get().runtime.output_dir
    log.info(f"Evasion results saved in {dir_path}).")
    with open(dir_path+"/cmh_attack.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    #test changes 
    log.info(f"Old Neighboors of {target_node}: {env.original_graph.neighbors(target_node)}")
    log.info(f"New Neighboors of {target_node}: {new_graph.neighbors(target_node)}")




if __name__ == "__main__":
    main()
