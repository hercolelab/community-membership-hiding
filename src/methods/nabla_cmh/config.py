"""
Config file for the nabla_cmh method.
We store all the hyperparameters in a dictionary here.
"""

from typing import Tuple, List

HYPERPARAMETERS = {
    "kar": {
        "training_greedy": {
            "testing_greedy": {
                "promising_action_coeffs": [0.713385915481054, 0.1346800221052267, 0.10264636655226009, 0.049287695861459285],
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 60, "lr": 0.0982, "lambd": 0.798},
                    "betaFactor_1": {"T": 100, "lr": 0.22, "lambd": 1.73},
                    "betaFactor_2": {"T": 100, "lr": 0.237, "lambd": 3.436},
                },
            },
            "testing_louvain": {
                "promising_action_coeffs": [0.713385915481054, 0.1346800221052267, 0.10264636655226009, 0.049287695861459285],
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 100, "lr": 0.094, "lambd": 1.155},
                    "betaFactor_1": {"T": 80, "lr": 0.4, "lambd": 4.8},
                    "betaFactor_2": {"T": 80, "lr": 0.388, "lambd": 1.854},
                },
            },
            "testing_walktrap": {
                "promising_action_coeffs": [0.3333333333333333, 0.3333333333333333, 0.16666666666666666, 0.16666666666666666],
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 70, "lr": 0.17, "lambd": 0.699},
                    "betaFactor_1": {"T": 60, "lr": 0.3, "lambd": 3.77},
                    "betaFactor_2": {"T": 100, "lr": 0.479, "lambd": 4.429},
                },
            },
        }
    }
}

def get_hyperparams(
        dataset:str,
        train_alg:str,
        test_alg:str,
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
    test_alg : str
        The testing algorithm name
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

    hyps = HYPERPARAMETERS[dataset][f"training_{train_alg}"][f"testing_{test_alg}"][f"tau_{tau}"][f"betaFactor_{beta_factor}"]
    max_it = hyps["T"]
    lr = hyps["lr"]
    lambd = hyps["lambd"]
    return max_it, lr, lambd

def get_promising_action_coeffs(
        dataset:str,
        train_alg:str,
        test_alg:str,
        ) -> List[float]:
    """
    Get the promising actions coefficients for the nabla_cmh method.

    Parameters
    ----------
    dataset : str
        The dataset name
    train_alg : str 
        The training algorithm name
    test_alg = str
        The testing algorithm name
    
    Returns
    -------
    promising_action_coeffs : List[float]
        The promising action coefficients
    """
    return HYPERPARAMETERS[dataset][f"training_{train_alg}"][f"testing_{test_alg}"]["promising_action_coeffs"]