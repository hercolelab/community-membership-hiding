from src.utils.utils import DetectionAlgorithmsNames
from src.utils.utils import ExperimentHyps, iGraphRNG
from typing import List, Optional, Dict, Any
from src.community_detection.extra_algs.locale.locale import ig_leiden_locale
from src.community_detection.extra_algs.dgcluster.dgcluster import DGCluster
from src.community_detection.extra_algs.scd import ig_SCD
from cdlib import NodeClustering
import igraph as ig
import leidenalg as la


"""
Community detection algorithms using iGraph.

The algorithms are:
- Greedy
- Infomap
- Label Propagation
- Louvain
- Walktrap
- Leading Eigenvector
- Edge Betweenness
- Spin Glass
- Scalable Community Detection
- Leiden
- Locale
- DGCluster

"""

class CommunityDetectionAlg:
	"""Class for community detection algorithms using iGraph."""

	def __init__(self, alg_name: str, env, graph: Optional[ig.Graph] = None) -> None:
		"""
		Initialize the CommunityDetectionAlg object.

		Parameters
		----------
		alg_name : str
			The name of the algorithm.
		env : object
			The environment object containing graph metadata.
		graph : Optional[ig.Graph], default=None
			The graph to be used in the algorithm (optional).
		"""
		self.alg_name = alg_name
		self.graph = graph
		self.env = env
		self.seed = ExperimentHyps.seed.value

	def community_detection(self, graph: ig.Graph, args: Optional[Dict[str, Any]] = None) -> NodeClustering:
		"""
		Compute the community detection algorithm.

		Parameters
		----------
		graph : ig.Graph
			The graph to be analyzed.
		args : dict, optional
			The arguments for the algorithm.

		Returns
		-------
		NodeClustering
			NodeClustering object containing the detected communities.
		"""
		self.graph = graph
		da = DetectionAlgorithmsNames

		# Fix randomness of the community detection
		ig.set_random_number_generator(iGraphRNG())

		alg_map = {
			da.GRE.value: self.compute_gre,
			da.LOUV.value: self.compute_louv,
			da.WALK.value: self.compute_walk,
			da.LEID.value: self.compute_leid,
			da.INF.value: self.compute_inf,
			da.LAB.value: self.compute_lab,
			da.EIG.value: self.compute_eig,
			da.BTW.value: self.compute_btw,
			da.SPIN.value: self.compute_spin,
			da.SCD.value: self.compute_scd,
			da.LOC.value: self.compute_loc,
			da.DGC.value: self.compute_dgc,
		}

		if self.alg_name not in alg_map:
			raise ValueError(f"Invalid algorithm name: {self.alg_name}")
		return alg_map[self.alg_name](graph, args)

	def from_vertexcluster_tolist(self, communities: ig.VertexClustering) -> NodeClustering:
		"""
		Convert the VertexClustering object to a NodeClustering object.
		"""
		com_list = [c for c in communities]
		return NodeClustering(
			communities=com_list,
			graph=self.graph,
			method_name=self.alg_name,
			overlap=False
		)

	def compute_gre(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Greedy community detection algorithm.
		"""
		greed = graph.community_fastgreedy(**(args or {}))
		return self.from_vertexcluster_tolist(greed.as_clustering())

	def compute_inf(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Infomap community detection algorithm.
		"""
		infomap = graph.community_infomap(**(args or {}))
		return self.from_vertexcluster_tolist(infomap)

	def compute_lab(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Label Propagation community detection algorithm.
		"""
		lab = graph.community_label_propagation(**(args or {}))
		return self.from_vertexcluster_tolist(lab)

	def compute_louv(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Louvain community detection algorithm.
		"""
		louv = graph.community_multilevel(**(args or {}))
		return self.from_vertexcluster_tolist(louv)

	def compute_walk(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Walktrap community detection algorithm.
		"""
		walk = graph.community_walktrap(**(args or {}))
		return self.from_vertexcluster_tolist(walk.as_clustering())

	def compute_eig(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Leading Eigenvector community detection algorithm.
		"""
		eig = graph.community_leading_eigenvector(**(args or {}))
		return self.from_vertexcluster_tolist(eig)

	def compute_btw(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Edge Betweenness community detection algorithm.
		"""
		btw = graph.community_edge_betweenness(**(args or {}))
		return self.from_vertexcluster_tolist(btw.as_clustering())

	def compute_spin(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Spin Glass community detection algorithm.
		"""
		spin = graph.community_spinglass(**(args or {}))
		return self.from_vertexcluster_tolist(spin)

	def compute_leid(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Leiden community detection algorithm.
		"""
		leiden_args = {"seed": self.seed}
		if args:
			leiden_args.update(args)
		leid = la.find_partition(graph, la.ModularityVertexPartition, **leiden_args)
		return self.from_vertexcluster_tolist(leid)

	def compute_scd(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Scalable Community Detection algorithm.
		"""
		scd_args = {"iterations": 30}
		if args:
			scd_args.update(args)
		scd = ig_SCD(**scd_args)
		scd.fit(graph)
		return scd.get_memberships()

	def compute_loc(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the Leiden-LOCALE community detection algorithm.
		"""
		name = f"{self.env.graph_name}_{str(self.env.budget_multiplier).replace('.', '')}"
		loc = ig_leiden_locale(graph, name, **(args or {}))
		return loc

	def compute_dgc(self, graph: ig.Graph, args: Optional[Dict[str, Any]]) -> NodeClustering:
		"""
		Compute the DGCluster community detection algorithm.
		"""
		name = self.env.graph_name
		dgc = DGCluster(graph, name, **(args or {}))
		return dgc