"""
Config file for the nabla_cmh method.
We store all the hyperparameters in a dictionary here.
"""

from typing import Tuple, List

HYPERPARAMETERS = {
    "kar": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 150, "lr": 0.049896804988909554, "lambd": 0.6937132432994007,
                                       "promising_action_coeffs": [0.03065359, 0.0998102 , 0.26597402, 0.60356219]},
                    "betaFactor_1": {"T": 300, "lr": 0.05, "lambd": 1.718469437702412, 
                                     "promising_action_coeffs": [0.07139868, 0.61258643, 0.28165036, 0.03436453]},
                    "betaFactor_2": {"T": 100, "lr": 0.26517996186290144, "lambd": 1.811087686699454,
                                     "promising_action_coeffs": [0.84370771, 0.0388562 , 0.05513557, 0.06230051]},
            }
        }
    },

    "words": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 100, "lr": 0.05968677042101627, "lambd": 0.012261499898800966,
                                        "promising_action_coeffs": [0.67822063, 0.09332295, 0.07709143, 0.151365]},
                    "betaFactor_1": {"T": 110, "lr": 0.006926590971173929, "lambd": 0.04337678816503348,
                                      "promising_action_coeffs": [0.16845957, 0.26303619, 0.34527027, 0.22323397]},
                    "betaFactor_2": {"T": 120, "lr": 0.03449731948954638, "lambd": 0.27385206593494427,
                                      "promising_action_coeffs": [0.0432059 , 0.09101503, 0.26986606, 0.59591301]}
            }
        }
    },

    "vote": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 130, "lr": 0.01741404022500432, "lambd": 0.5963970307660161,
                                        "promising_action_coeffs": [0.05122234, 0.03213769, 0.64427686, 0.27236311]},
                    "betaFactor_1": {"T": 140, "lr": 0.01708833148323878, "lambd": 0.3705902649255662,
                                      "promising_action_coeffs": [0.48833039, 0.25428578, 0.01288531, 0.24449852]},
                    "betaFactor_2": {"T": 140, "lr": 0.011023306191490473, "lambd": 11.26928761550795,
                                      "promising_action_coeffs": [0.74531481, 0.07455831, 0.15811051, 0.02201636]}
            }
        }
    },

    "pow": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 110, "lr": 0.008424629753583818, "lambd": 11.584348234275256,
                                        "promising_action_coeffs": [0.13990988, 0.50692226, 0.10397525, 0.24919261]},
                    "betaFactor_1": {"T": 130, "lr": 0.008407302218656706, "lambd": 18.134958318370664,
                                      "promising_action_coeffs": [0.05047794, 0.17979355, 0.4181838 , 0.35154471]},
                    "betaFactor_2": {"T": 130, "lr": 0.007429111691215709, "lambd": 37.13450949970305,
                                      "promising_action_coeffs": [0.0113813 , 0.90502937, 0.06155817, 0.02203116]}
            }
        }
    },

    "fb-75": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 80, "lr": 0.005095090879146563, "lambd": 21.483355166521605,
                                        "promising_action_coeffs": [0.09241378, 0.3651285 , 0.4648003 , 0.07765742]},
                    "betaFactor_1": {"T": 140, "lr": 0.004572137442171874, "lambd": 0.158912754967332,
                                      "promising_action_coeffs": [0.29490579, 0.59597709, 0.09845179, 0.01066533]},
                    "betaFactor_2": {"T": 140, "lr": 0.00928855144862304, "lambd": 0.6074906655014723,
                                      "promising_action_coeffs": [0.24062604, 0.09487533, 0.55131609, 0.11318254]}
            }
        }
    },

    "cond-mat": {
        "training_greedy": {
            "tau_0.5": {
                    "betaFactor_0.5": {"T": 150, "lr": 0.003567160381067974, "lambd": 2.569501161861668,
                                        "promising_action_coeffs": [0.41053921, 0.05129182, 0.15335925, 0.38480971]},
                    "betaFactor_1": {"T": 140, "lr": 0.0010500793424532866, "lambd": 17.289349336336716,
                                      "promising_action_coeffs": [0.40392329, 0.21622696, 0.05229272, 0.32755703]},
                    "betaFactor_2": {"T": 90, "lr": 0.0017399212257003674, "lambd": 0.2876713350614822,
                                      "promising_action_coeffs": [0.08542925, 0.58998522, 0.12063156, 0.20395397]}
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