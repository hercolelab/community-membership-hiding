from src.graph_environment.env import GraphEnvironment
from src.utils.utils import ExperimentHyps

# ------ Example usage ------ #
graph_name = "KAR"
community_detection_alg = "LOUV"
env = GraphEnvironment(graph_name, community_detection_alg)
graph = env.original_graph.copy()
community = env.target_community
print(community)
subgraph = graph.subgraph(community)
print(subgraph.summary())