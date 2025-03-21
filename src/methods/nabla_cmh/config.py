"""
Config file for the nabla_cmh method.
We store all the hyperparameters in a dictionary here.
"""

from typing import Tuple, List

HYPERPARAMETERS = {
    "kar": {
        "training_greedy": {
            "testing_greedy": {
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 60, "lr": 0.0982, "lambd": 0.798,
                                       "promising_action_coeffs": [0.057360814315329266, 0.3194249576577513, 0.09358665774472155, 0.5296275702821979]},
                    "betaFactor_1": {"T": 140, "lr": 0.07638556181631337, "lambd": 0.4179266521576686, 
                                     "promising_action_coeffs": [0.057360814315329266, 0.3194249576577513, 0.09358665774472155, 0.5296275702821979]},
                    "betaFactor_2": {"T": 100, "lr": 0.237, "lambd": 3.436,
                                     "promising_action_coeffs": [0.057360814315329266, 0.3194249576577513, 0.09358665774472155, 0.5296275702821979]},
                },
            },
            "testing_louvain": {
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 100, "lr": 0.094, "lambd": 1.155,
                                       "promising_action_coeffs": [0.713385915481054, 0.1346800221052267, 0.10264636655226009, 0.049287695861459285]},
                    "betaFactor_1": {"T": 130, "lr": 0.37539138110922154, "lambd": 4.1475097464280015,
                                     "promising_action_coeffs": [0.12203004345667687, 0.5988271187863038, 0.07296875674933952, 0.2061740810076798]},
                    "betaFactor_2": {"T": 80, "lr": 0.388, "lambd": 1.854,
                                     "promising_action_coeffs": [0.713385915481054, 0.1346800221052267, 0.10264636655226009, 0.049287695861459285]},
                },
            },
            "testing_walktrap": {
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 70, "lr": 0.17, "lambd": 0.699,
                                       "promising_action_coeffs": [0.3333333333333333, 0.3333333333333333, 0.16666666666666666, 0.16666666666666666]},
                    "betaFactor_1": {"T": 100, "lr": 0.3314628680780704, "lambd": 2.3574811413312085,
                                     "promising_action_coeffs": [0.42920813982593226, 0.32531970596788634, 0.18228455132736981, 0.06318760287881156]},
                    "betaFactor_2": {"T": 100, "lr": 0.479, "lambd": 4.429,
                                     "promising_action_coeffs": [0.3333333333333333, 0.3333333333333333, 0.16666666666666666, 0.16666666666666666]},
                },
            },
        }
    },

    "words": {
        "training_greedy": {
            "testing_greedy": {
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 60, "lr": 0.0549, "lambd": 0.543,
                                        "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]},
                    "betaFactor_1": {"T": 90, "lr": 0.012, "lambd": 0.827,
                                      "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]},
                    "betaFactor_2": {"T": 50, "lr": 0.01, "lambd": 1.225,
                                      "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]}
                }
            },
            "testing_louvain": {
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 60, "lr": 0.0681, "lambd": 0.5428,
                                       "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]},
                    "betaFactor_1": {"T": 100, "lr": 0.0678, "lambd": 1.678,
                                     "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]},
                    "betaFactor_2": {"T": 70, "lr": 0.169, "lambd": 1.318,
                                     "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]}
                }
            },
            "testing_walktrap": {
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 90, "lr": 0.0827, "lambd": 0.456,
                                        "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]},
                    "betaFactor_1": {"T": 70, "lr": 0.0482, "lambd": 0.556,
                                      "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]},
                    "betaFactor_2": {"T": 90, "lr": 0.1817, "lambd": 1.18, 
                                     "promising_action_coeffs": [0.0004028761034536307, 0.053901189311128125, 0.08197490788515284, 0.8637210267002654]}
                }
            }
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
    coeffs = hyps["promising_action_coeffs"]
    return max_it, lr, lambd, coeffs