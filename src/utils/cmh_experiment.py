from src.utils.utils import ExperimentHyps, Utils, DRL_agentHyps, FilePaths
from src.graph_environment.env import GraphEnvironment
from src.utils.utils import EvasionAlgorithmsNames
from src.baselines.random import RandomHiding
from src.baselines.degree import DegreeHiding
from src.baselines.betweenness import CentralityHiding
from src.baselines.roam import RoamHiding
from src.baselines.dice import DiceHiding
from src.baselines.clustering import ClusteringHiding
from src.baselines.triad_breaking import TriadBreakingHiding
from src.methods.nabla_cmh.nabla_cmh import nablaCMH
from src.methods.nabla_cmh_projection.nabla_cmh_proj import nablaCMH_proj
from src.methods.drl_agent.agent import Agent
from typing import List, Optional, Dict, Any
import logging
from time import time
import json
from hydra.core.hydra_config import HydraConfig
from tqdm import trange
import numpy as np

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
    ) -> None:
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
        self.dir_paths: List[str] = []

    # ============================================================================= #
    #                                RESET FUNCTION                                 #
    # ============================================================================= #

    def set_parameters(self, beta_factor: float, tau: float) -> None:
        """
        Set the parameters of the CMH problem on the environment.

        Parameters
        ----------
        beta_factor: float
            Budget multiplier.
        tau: float
            Similarity threshold.
        """
        self.env.budget_multiplier = beta_factor
        self.env.tau = tau
        self.env.budget = self.env.get_budget()
        self.budget = self.env.budget
        self.dir_paths = []
        for detection_alg in self.env.community_detection_alg_names_output:
            dir_path = (
                f"{HydraConfig.get().runtime.output_dir}/"
                f"{self.env.graph_name_output}/"
                f"{detection_alg}/"
                f"tau_{self.env.tau}/"
                f"betaFactor_{self.env.budget_multiplier}/json_results/"
            )
            Utils.check_dir(dir_path)
            self.dir_paths.append(dir_path)

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
        self.set_results_dict()

        sizes = trange(len(preferred_sizes), desc="* * * Community Step", leave=True)
        for i in sizes:
            self.env.preferred_community_size = preferred_sizes[i]
            self.env.change_target_community()
            experiment_steps = len(self.env.list_target_nodes)
            sizes.set_description(f"* * * Community Step {i+1}/{len(preferred_sizes)}")
            exp_steps = trange(experiment_steps, desc="* * * Node Step", leave=False)

            for j in exp_steps:
                self.env.change_target_node()
                target_node = self.env.target_node

                for alg in self.evasion_algs:
                    exp_steps.set_description(
                        f"* * * Node Step {j+1}/{experiment_steps} | {alg}"
                    )
                    evasion_alg = self._get_evasion_algorithm(alg, target_node)
                    func_call = self._get_func_call(alg, evasion_alg)

                    start = time()
                    result = func_call()
                    end = time()
                    total_time = end - start

                    # Unpack variables based on algorithm type
                    if alg == EvasionAlgorithmsNames.NABLA.name:
                        new_graph, steps, changes, add_results = result
                    elif alg == EvasionAlgorithmsNames.NABLAP.name:
                        new_graph, steps, changes, add_results = result
                    else:
                        new_graph, steps, changes = result
                        add_results = None

                    goals, nmis = self.env.get_metrics(new_graph)
                    self.save_results(
                        alg, target_node, steps, changes, goals, nmis, total_time, add_results
                    )

    def _get_evasion_algorithm(self, alg: str, target_node: int) -> Any:
        """
        Factory method to instantiate the correct evasion algorithm.
        """
        if alg == EvasionAlgorithmsNames.RAND.name:
            return RandomHiding(self.env, target_node, self.budget)
        elif alg == EvasionAlgorithmsNames.DEG.name:
            return DegreeHiding(self.env, target_node, self.budget)
        elif alg == EvasionAlgorithmsNames.BETW.name:
            return CentralityHiding(self.env, target_node, self.budget)
        elif alg == EvasionAlgorithmsNames.ROAM.name:
            return RoamHiding(self.env, target_node, self.budget)
        elif alg == EvasionAlgorithmsNames.DICE.name:
            return DiceHiding(self.env, target_node, self.budget)
        elif alg == EvasionAlgorithmsNames.NABLA.name:
            return nablaCMH(self.env, target_node, self.env.budget)
        elif alg == EvasionAlgorithmsNames.NABLAP.name:
            return nablaCMH_proj(self.env, target_node, self.env.budget)
        elif alg == EvasionAlgorithmsNames.CLU.name:
            return ClusteringHiding(self.env, target_node, self.budget)
        elif alg == EvasionAlgorithmsNames.TRI.name:
            return TriadBreakingHiding(self.env, target_node, self.budget)
        elif alg == EvasionAlgorithmsNames.DRL.name:
            return Agent(env=self.env)
        else:
            raise ValueError(f"Invalid evasion attack algorithm: {alg}")

    def _get_func_call(self, alg: str, evasion_alg: Any):
        """
        Returns the function to call for the given algorithm.
        """
        if alg == EvasionAlgorithmsNames.DRL.name:
            return lambda: evasion_alg.test(
                lr=DRL_agentHyps.LR_EVAL.value,
                gamma=DRL_agentHyps.GAMMA_EVAL.value,
                lambda_metric=DRL_agentHyps.LAMBDA_EVAL.value,
                alpha_metric=DRL_agentHyps.ALPHA_EVAL.value,
                epsilon_prob=DRL_agentHyps.EPSILON_EVAL.value,
                model_path=FilePaths.TRAINED_MODEL.value,
            )
        else:
            return lambda: evasion_alg.community_membership_hiding()

    # ============================================================================= #
    #                                  LOG FUNCTIONS                                #
    # ============================================================================= #

    def set_results_dict(self) -> None:
        """
        Set the results dictionaries to store the results of the experiment.
        """
        for dir_path, detection_alg_name in zip(self.dir_paths, self.env.community_detection_alg_names_output):
            for alg in self.evasion_algs:
                filename = f"{getattr(EvasionAlgorithmsNames, alg).value}.json"
                results = {
                    "graph": self.env.graph_name_output,
                    "community_detection": detection_alg_name,
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
                with open(dir_path + filename, "w") as json_file:
                    json.dump(results, json_file, indent=4, cls=NumpyEncoder)
                # Additional json for nabla-cmh
                if alg == EvasionAlgorithmsNames.NABLA.name:
                    filename_additional = f"{getattr(EvasionAlgorithmsNames, alg).value}_additional.json"
                    results_additional = {"hidings": []}
                    with open(dir_path + filename_additional, "w") as json_file:
                        json.dump(results_additional, json_file, indent=4, cls=NumpyEncoder)

    def save_results(
        self,
        alg: str,
        target_node: int,
        steps: int,
        changes: dict,
        goals: List[int],
        nmis: List[float],
        time: float,
        additional_results: Optional[dict],
    ) -> None:
        """
        Save the results of a single evasion algorithm for all detection algs in a json file.

        Parameters
        ----------
        alg: str
            Name of the evasion algorithm.
        target_node: int
            Target node to hide.
        steps: int
            Number of steps done to reach the goal.
        changes: dict
            Changes done to the graph.
        goals: List[int]
            Goal reached by the evasion algorithm for each detection algorithm.
        nmis: List[float]
            Normalized Mutual Information for each detection algorithm.
        time: float
            Time to run the evasion algorithm.
        additional_results: Optional[dict]
            Additional results for nabla-cmh.
        """
        filename = f"{getattr(EvasionAlgorithmsNames, alg).value}.json"
        com_size = self.env.target_community_size
        for idx, dir_path in enumerate(self.dir_paths):
            with open(dir_path + filename, "r") as json_file:
                results = json.load(json_file)
            results["target_node"].append(target_node)
            results["community_size"].append(com_size)
            results["steps"].append(steps)
            results["changes"].append(changes)
            results["goal"].append(goals[idx])
            results["nmi"].append(nmis[idx])
            results["time"].append(time)
            with open(dir_path + filename, "w") as json_file:
                json.dump(convert(results), json_file, indent=4, cls=NumpyEncoder)
            if additional_results is not None:
                filename_additional = f"{getattr(EvasionAlgorithmsNames, alg).value}_additional.json"
                with open(dir_path + filename_additional, "r") as json_file:
                    results_additional = json.load(json_file)
                results_additional["hidings"].append(additional_results)
                with open(dir_path + filename_additional, "w") as json_file:
                    json.dump(convert(results_additional), json_file, indent=4, cls=NumpyEncoder)

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)





