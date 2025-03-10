from src.graph_environment.env import GraphEnvironment
from src.utils.utils import ExperimentHyps

# ------ Example usage ------ #
graph_name = "FB_75"
community_detection_alg = "LOUV"
env = GraphEnvironment(graph_name, community_detection_alg)
env.print_environment_info()