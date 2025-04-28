from src.utils.utils import ExperimentHyps, Utils, DRL_agentHyps, FilePaths
from src.graph_environment.env import GraphEnvironment
from src.utils.utils import EvasionAlgorithmsNames
from src.baselines.random import RandomHiding
from src.baselines.degree import DegreeHiding
from src.baselines.betweenness import CentralityHiding
from src.baselines.roam import RoamHiding
from src.baselines.dice import DiceHiding
from src.methods.nabla_cmh.nabla_cmh import nablaCMH
from src.methods.drl_agent.agent import Agent
from typing import List, Optional
import logging
from time import time
import json
from hydra.core.hydra_config import HydraConfig
from rich.progress import Progress

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
        

        


    # ============================================================================= #
    #                                RESET FUNCTION                                 #
    # ============================================================================= #
        

    def set_parameters(self, beta_factor:float, tau:float)-> None:
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
        self.dir_paths: List[str] = []
        for detection_alg in self.env.community_detection_alg_names_output:
            dir_path: str = HydraConfig.get().runtime.output_dir + f"/{self.env.graph_name_output}" + f"/{detection_alg}" + f"/tau_{self.env.tau}" + f"/betaFactor_{self.env.budget_multiplier}" + "/json_results/"
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

        # Set the dictionaries to store results for each algorithm
        self.set_results_dict()

        with Progress() as progress:
            outer_task = progress.add_task("[cyan]Community Steps...", total=3)
            inner_task = None

            for i in range(len(preferred_sizes)):

                progress.update(
                    outer_task,
                    description=f"[cyan] Community Step {i+1}/3"
                )
                # Set the preferred community size to the environment
                self.env.preferred_community_size = preferred_sizes[i]
                # Change the target community
                self.env.change_target_community()
                experiment_steps = len(self.env.list_target_nodes)
                
                inner_task = progress.add_task(
                    f"[green]Processing Nodes for Community {i+1}...",
                    total=experiment_steps
                )   
                for j in range(experiment_steps):
                    # Change target node within the target community
                    self.env.change_target_node()
                    target_node = self.env.target_node
                    progress.update(
                        inner_task,
                        description=f"[green] Node Step {j+1}/{experiment_steps}"
                    )

                    # Run the evasion algorithms
                    for alg in self.evasion_algs:
                        progress.update(
                            inner_task,
                            description=f"[green] Node Step {j+1}/{experiment_steps} - {alg.upper()}"
                        )
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
                        elif alg == EvasionAlgorithmsNames.NABLA.name:
                            evasion_alg = nablaCMH(self.env, target_node, self.env.budget)
                        elif alg == EvasionAlgorithmsNames.DRL.name:
                            evasion_alg = Agent(env=self.env)
                        else:
                            raise ValueError("Invalid evasion attack algorithm")
                        
                        # Set the hiding function
                        if alg == EvasionAlgorithmsNames.DRL.name:
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
                        goals, nmis = self.env.get_metrics(new_graph)

                        # Save results
                        if alg == EvasionAlgorithmsNames.NABLA.name:
                            self.save_results(alg, target_node, steps, changes, goals, nmis, total_time, add_results)
                        else:
                            self.save_results(alg, target_node, steps, changes, goals, nmis, total_time, None)
                    
                    progress.update(inner_task, advance=1)
                progress.update(
                    inner_task,
                    description=f"[green] Node Step {j+1}/{experiment_steps}"
                )
                progress.update(outer_task, advance=1)


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
                    json.dump(results, json_file, indent=4)
                # We add an additional json file for the additional results of the nabla-cmh algorithm (if any)    
                if alg == EvasionAlgorithmsNames.NABLA.name:
                    filename_additional = f"{getattr(EvasionAlgorithmsNames, alg).value}_additional.json"
                    results_additional = {
                    "hidings": []
                    }
                    with open(dir_path + filename_additional, "w") as json_file:
                        json.dump(results_additional, json_file, indent=4)

    
    def save_results(
            self,
            alg: str,
            target_node: int,
            steps: int,
            changes: dict,
            goals : List[int],
            nmis: List[float],
            time: float,
            additional_results: Optional[dict]) -> None:

        """
        Save the results of a single evasion algorithm for all detection algs in a json file.

        Parameters
        ----------
        alg: str
            Name of the evasion algorithm.
        target_node: int
            Target node to hide.
        target_community_size: int
            Size of the target community.
        steps: int
            Number of steps done to reach the goal.
        changes: dict
            Changes done to the graph.
        goal: int
            Goal reached by the evasion algorithm.
        nmi: float
            Normalized Mutual Information between the original and the counterfactual community structure.
        time: float
            Time to run the evasion algorithm.
        """

        filename: str = f"{getattr(EvasionAlgorithmsNames, alg).value}.json"
        com_size: int = self.env.target_community_size
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
                json.dump(results, json_file, indent=4)
            
            if additional_results is not None:
                filename_additional: str = f"{getattr(EvasionAlgorithmsNames, alg).value}_additional.json"
                with open(dir_path + filename_additional, "r") as json_file:
                    results_additional = json.load(json_file)
                results_additional["hidings"].append(additional_results)
                with open(dir_path + filename_additional, "w") as json_file:
                    json.dump(results_additional, json_file, indent=4)
        

        


