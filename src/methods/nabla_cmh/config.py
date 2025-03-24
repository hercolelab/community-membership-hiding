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
                    "betaFactor_0.5": {"T": 50, "lr": 0.05678728204440248, "lambd": 0.24554341302058932,
                                        "promising_action_coeffs": [0.2153310085656631, 0.24476940435231107, 0.0604602986930444, 0.47943928838898137]},
                    "betaFactor_1": {"T": 150, "lr": 0.07687138898896992, "lambd": 4.037970428614292,
                                      "promising_action_coeffs": [0.4180580631911592, 0.31335952062755795, 0.17243565485777484, 0.09614676132350794]},
                    "betaFactor_2": {"T": 90, "lr": 0.01822857030151786, "lambd": 1.764914288528197,
                                      "promising_action_coeffs": [0.23741748074475236, 0.3323198665401349, 0.35854601751800896, 0.07171663519710361]}
                }
            },
            "testing_louvain": {
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 110, "lr": 0.0971743377120408, "lambd": 0.9667710599251506,
                                       "promising_action_coeffs": [0.6057974123839814, 0.03466084170568882, 0.046851933588303724, 0.31268981232202586]},
                    "betaFactor_1": {"T": 110, "lr": 0.07344665099594508, "lambd": 3.175333768101549,
                                     "promising_action_coeffs": [0.18461527488452079, 0.612937096442619, 0.08037548951136898, 0.1220721391614912]},
                    "betaFactor_2": {"T": 120, "lr": 0.18381116539675177, "lambd": 1.9494612145529215,
                                     "promising_action_coeffs": [0.039289324202547715, 0.4452353738461994, 0.4555472601188823, 0.05992804183237058]}
                }
            },
            "testing_walktrap": {
                "tau_0.5": {
                    "betaFactor_0.5": {"T": 150, "lr": 0.08672128332266157, "lambd": 0.5355398319386634,
                                        "promising_action_coeffs": [0.2515388443854097, 0.1568254792841614, 0.01301050113435496, 0.5786251751960738]},
                    "betaFactor_1": {"T": 120, "lr": 0.10528696084946695, "lambd": 0.9781951689845612,
                                      "promising_action_coeffs": [0.1722583776943468, 0.3563689369466441, 0.009741577843741165, 0.4616311075152679]},
                    "betaFactor_2": {"T": 150, "lr": 0.17754671443948764, "lambd": 0.08299581556361141, 
                                     "promising_action_coeffs": [0.20044991207447863, 0.11239547886172739, 0.20067787615053434, 0.48647673291325966]}
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