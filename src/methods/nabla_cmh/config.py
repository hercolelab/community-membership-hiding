"""
Config file for the nabla_cmh method.
We store all the hyperparameters in a dictionary here.
"""

from typing import Tuple, List

HYPERPARAMETERS = {
    "kar": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 60, "lr": 0.0982, "lambd": 0.798,
                                       "promising_action_coeffs": [0.057360814315329266, 0.3194249576577513, 0.09358665774472155, 0.5296275702821979]},
                    "betaFactor_1": {"T": 140, "lr": 0.07638556181631337, "lambd": 0.4179266521576686, 
                                     "promising_action_coeffs": [0.057360814315329266, 0.3194249576577513, 0.09358665774472155, 0.5296275702821979]},
                    "betaFactor_2": {"T": 100, "lr": 0.237, "lambd": 3.436,
                                     "promising_action_coeffs": [0.057360814315329266, 0.3194249576577513, 0.09358665774472155, 0.5296275702821979]},
            }
        }
    },

    "words": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 50, "lr": 0.05678728204440248, "lambd": 0.24554341302058932,
                                        "promising_action_coeffs": [0.2153310085656631, 0.24476940435231107, 0.0604602986930444, 0.47943928838898137]},
                    "betaFactor_1": {"T": 150, "lr": 0.07687138898896992, "lambd": 4.037970428614292,
                                      "promising_action_coeffs": [0.4180580631911592, 0.31335952062755795, 0.17243565485777484, 0.09614676132350794]},
                    "betaFactor_2": {"T": 90, "lr": 0.01822857030151786, "lambd": 1.764914288528197,
                                      "promising_action_coeffs": [0.23741748074475236, 0.3323198665401349, 0.35854601751800896, 0.07171663519710361]}
            }
        }
    },

    "vote": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 50, "lr": 0.05678728204440248, "lambd": 0.24554341302058932,
                                        "promising_action_coeffs": [0.2153310085656631, 0.24476940435231107, 0.0604602986930444, 0.47943928838898137]},
                    "betaFactor_1": {"T": 150, "lr": 0.07687138898896992, "lambd": 4.037970428614292,
                                      "promising_action_coeffs": [0.4180580631911592, 0.31335952062755795, 0.17243565485777484, 0.09614676132350794]},
                    "betaFactor_2": {"T": 90, "lr": 0.01822857030151786, "lambd": 1.764914288528197,
                                      "promising_action_coeffs": [0.23741748074475236, 0.3323198665401349, 0.35854601751800896, 0.07171663519710361]}
            }
        }
    },

    "pow": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 50, "lr": 0.05678728204440248, "lambd": 0.24554341302058932,
                                        "promising_action_coeffs": [0.2153310085656631, 0.24476940435231107, 0.0604602986930444, 0.47943928838898137]},
                    "betaFactor_1": {"T": 150, "lr": 0.07687138898896992, "lambd": 4.037970428614292,
                                      "promising_action_coeffs": [0.4180580631911592, 0.31335952062755795, 0.17243565485777484, 0.09614676132350794]},
                    "betaFactor_2": {"T": 90, "lr": 0.01822857030151786, "lambd": 1.764914288528197,
                                      "promising_action_coeffs": [0.23741748074475236, 0.3323198665401349, 0.35854601751800896, 0.07171663519710361]}
            }
        }
    },

    "fb-75": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 50, "lr": 0.05678728204440248, "lambd": 0.24554341302058932,
                                        "promising_action_coeffs": [0.2153310085656631, 0.24476940435231107, 0.0604602986930444, 0.47943928838898137]},
                    "betaFactor_1": {"T": 150, "lr": 0.07687138898896992, "lambd": 4.037970428614292,
                                      "promising_action_coeffs": [0.4180580631911592, 0.31335952062755795, 0.17243565485777484, 0.09614676132350794]},
                    "betaFactor_2": {"T": 90, "lr": 0.01822857030151786, "lambd": 1.764914288528197,
                                      "promising_action_coeffs": [0.23741748074475236, 0.3323198665401349, 0.35854601751800896, 0.07171663519710361]}
            }
        }
    }
}

def get_hyperparams(
        dataset:str,
        train_alg:str,
        tau:float,
        beta_factor:float,
        ) -> Tuple[int, float, float]:
    """
    Get the hyperparameters for the nabla_cmh method.

    Parameters
    ----------
    dataset : str
        The dataset name
    train_alg : str
        The training algorithm name
    tau : float
        The similarity threshold
    beta_factor : float
        The budget multiplier

    Returns
    -------
    max_it : int
        The number of max iterations
    lr : float
        The learning rate for the optimization process
    lambd: float
        The regularization constant for the loss
    """

    hyps = HYPERPARAMETERS[dataset][f"training_{train_alg}"][f"tau_{tau}"][f"betaFactor_{beta_factor}"]
    max_it = hyps["T"]
    lr = hyps["lr"]
    lambd = hyps["lambd"]
    coeffs = hyps["promising_action_coeffs"]
    return max_it, lr, lambd, coeffs