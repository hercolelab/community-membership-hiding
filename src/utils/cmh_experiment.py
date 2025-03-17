from src.utils.utils import ExperimentHyps, Utils, FilePaths
from src.graph_environment.env import GraphEnvironment
from src.utils.utils import EvasionAlgorithmsNames
from omegaconf import DictConfig
from src.baselines.random import RandomHiding
from src.baselines.degree import DegreeHiding
from src.baselines.betweenness import CentralityHiding
from src.baselines.roam import RoamHiding
from src.baselines.dice import DiceHiding
from typing import List, Callable, Tuple
from omegaconf import DictConfig
from tqdm import trange
import multiprocessing
import igraph as ig
import copy
import random
import logging
import hydra
import yaml
from time import time
import json
from hydra.core.hydra_config import HydraConfig


log = logging.getLogger(__name__)
class CmhExperiment:
    """
    Class to evaluate the performance of several evasion algorithms 
    for the Community Membership Hiding problem.
    """

    def __init__(
            self,
            evasion_algs: List[str],
            env: GraphEnvironment,
            )-> None:
        """
        Constructor of the class.

        Parameters
        ----------
        evasion_algs: List[str]
            List of evasion algorithms to evaluate.
            
        env: GraphEnvironment
            Graph environment to evaluate the evasion algorithms.
        """

        self.evasion_algs: List[str] = evasion_algs
        self.env: GraphEnvironment = env
        self.budget: int = self.env.budget
        self.dir_path: str = HydraConfig.get().runtime.output_dir + f"/{self.env.graph_name_output}" + f"/{self.env.community_detection_alg_name_output}" + f"/tau_{self.env.tau}" + f"/betaFactor_{self.env.budget_multiplier}" + "/json_results/"
        Utils.check_dir(self.dir_path)


    # ============================================================================= #
    #                                RESET FUNCTION                                 #
    # ============================================================================= #
        

    def set_parameters(
            self,
            beta_factor:float,
            tau:float,
        )-> None:

        """
        Set the parameters of the CMH problem
        """
        
        self.env.budget_multiplier = beta_factor
        self.env.tau = tau
        self.env.budget = self.env.get_budget()
        self.budget = self.env.budget


    # ============================================================================= #
    #                              EXPERIMENT FUNCTION                              #
    # ============================================================================= #

    def run_experiment(self) -> None:
        """
        Function to run the experiment for the CMH problem.
        The experiment is made as follows:
            - we pick a community with size close to the three preferred sizes (0.2, 0.5, 0.8 of the maximum community size)
            - we select at most 100 nodes from it
            - for each node we run the evasion algorithms
        """
        
        preferred_sizes = ExperimentHyps.target_community_size.value
        max_experiment_steps = ExperimentHyps.max_steps_community_eval.value

        # Set the dictionaries to store results for each algorithm
        self.set_results_dict()

        for i in range(len(preferred_sizes)):
            # Set the preferred community size to the environment
            self.env.preferred_community_size = preferred_sizes[i]
            # Change the target community
            self.env.change_target_community()
            experiment_steps = min(max_experiment_steps, len(self.env.target_community))

            for j in range(experiment_steps):
                # Change target node within the target community
                self.env.change_target_node(step=j)
                target_node = self.env.target_node

                # Run the evasion algorithms
                for alg in self.evasion_algs:
                    if alg == EvasionAlgorithmsNames.RAND.name:
                        evasion_alg = RandomHiding(self.env, target_node, self.budget)
                    elif alg == EvasionAlgorithmsNames.DEG.name:
                        evasion_alg = DegreeHiding(self.env, target_node, self.budget)
                    elif alg == EvasionAlgorithmsNames.BETW.name:
                        evasion_alg = CentralityHiding(self.env, target_node, self.budget)
                    elif alg == EvasionAlgorithmsNames.ROAM.name:
                        evasion_alg = RoamHiding(self.env, target_node, self.budget)
                    elif alg == EvasionAlgorithmsNames.DICE.name:
                        evasion_alg = DiceHiding(self.env, target_node, self.budget)
                    else:
                        raise ValueError("Invalid evasion attack algorithm")
                    
                    log.info(f"{i+1}-st Community size: {self.env.target_community_size} | Testing episode {j+1} |  Evasion algorithm: {alg}")
                    
                    # Counterfactual graph
                    start = time()
                    new_graph, steps, changes = evasion_alg.community_membership_hiding()
                    end = time()
                    total_time = end - start

                    # Compute metrics
                    goal, nmi = self.env.get_metrics(new_graph)

                    # Save results
                    self.save_results(alg, target_node, self.env.target_community_size, steps, changes, goal, nmi, total_time)


    # ============================================================================= #
    #                                  LOG FUNCTIONS                                #
    # ============================================================================= #

    def set_results_dict(self) -> None:
        """
        Set the results dictionaries to store the results of the experiment.
        """

        for alg in self.evasion_algs:
            filename = f"{getattr(EvasionAlgorithmsNames, alg).value}.json"
            results = {
                "graph": self.env.graph_name_output,
                "community_detection": self.env.community_detection_alg_name_output,
                "beta_factor": self.env.budget_multiplier,
                "tau": self.env.tau,
                "budget": self.budget,
                "target_node": [],
                "community_size": [],
                "steps": [],
                "changes": [],
                "goal": [],
                "nmi": [],
                "time": [],
            }
            with open(self.dir_path + filename, "w") as json_file:
                json.dump(results, json_file, indent=4)
    
    def save_results(
            self,
            alg: str,
            target_node: int,
            target_community_size: int,
            steps: int,
            changes: dict,
            goal : int,
            nmi: float,
            time: float,
        ) -> None:

        """
        Save the results of a single evasion algorithm in a json file.
        """

        filename: str = f"{getattr(EvasionAlgorithmsNames, alg).value}.json"
        with open(self.dir_path + filename, "r") as json_file:
            results = json.load(json_file)
        
        results["target_node"].append(target_node)
        results["community_size"].append(target_community_size)
        results["steps"].append(steps)
        results["changes"].append(changes)
        results["goal"].append(goal)
        results["nmi"].append(nmi)
        results["time"].append(time)

        with open(self.dir_path + filename, "w") as json_file:
            json.dump(results, json_file, indent=4)
        

        


