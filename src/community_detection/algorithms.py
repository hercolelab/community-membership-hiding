from src.utils.utils import DetectionAlgorithmsNames
from src.utils.utils import ExperimentHyps, iGraphRNG
from typing import List, Optional
from cdlib import algorithms
import cdlib
import os
import igraph as ig
import random


"""
Community detection algorithms using iGraph.

The algorithms are:
- Greedy
- Infomap
- Label Propagation
- Louvain
- Walktrap


"""

class CommunityDetectionAlg(object):
	"""Class for the community detection algorithms using iGraph"""

	def __init__(self, alg_name: str, graph: Optional[ig.Graph]=None) -> None:
		"""
		Initialize the DetectionAlgorithm object

		Parameters
		----------
		alg_name : str
			The name of the algorithm
		graph : Optional[ig.Graph], default=None
			The graph to be used in the algorithm (optional)
		"""
		self.alg_name = alg_name
		self.graph = graph
		self.seed = ExperimentHyps.seed.value

	def community_detection(self, graph: ig.Graph, args: dict = None) -> cdlib.NodeClustering:
		"""
		Compute the community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be analyzed
		args : dict
			The arguments for the algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""

		self.graph = graph
		da = DetectionAlgorithmsNames
		
		# Fix Randomness of the community detection
		custom_rng = iGraphRNG()
		ig.set_random_number_generator(custom_rng)

		# Choose the algorithm
		if self.alg_name == da.GRE.value:
			return self.compute_gre(graph, args)
		elif self.alg_name == da.LOUV.value:
				return self.compute_louv(graph, args)
		elif self.alg_name == da.WALK.value:
			return self.compute_walk(graph, args)	
		elif self.alg_name == da.INF.value:
			return self.compute_inf(graph, args)
		elif self.alg_name == da.LAB.value:
			return self.compute_lab(graph, args)
		elif self.alg_name == da.EIG.value:
			return self.compute_eig(graph, args)
		elif self.alg_name == da.BTW.value:
			return self.compute_btw(graph, args)
		elif self.alg_name == da.SPIN.value:
			return self.compute_spin(graph, args)
		else:
			raise ValueError("Invalid algorithm name")
		
	def from_vertexcluster_tolist(self, communities: ig.VertexClustering) -> cdlib.NodeClustering:
		"""
		Convert the VertexClustering object to a list of list of vertices
		"""
		com_list = [c for c in communities]
		# Create a NodeClustering object
		node_cluster = cdlib.NodeClustering(
			communities=com_list,
			graph=self.graph,
			method_name=self.alg_name,
			overlap=False
		)
		return node_cluster
	
	def compute_gre(self, graph: ig.Graph, args_gre: dict) -> List[List[int]]:
		"""
		Compute the Greedy community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be clustered
		args_greed : dict
			The arguments for the Greedy algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""

		if args_gre is None:
			greed = graph.community_fastgreedy()
		else:
			greed = graph.community_fastgreedy(**args_gre)
		# Need to be converted to VertexClustering object -> as_clustering() method
		return self.from_vertexcluster_tolist(greed.as_clustering())

	def compute_inf(self, graph: ig.Graph, args_infomap: dict) -> List[List[int]]:
		"""
		Compute the Infomap community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be clustered
		args_infomap : dict
			The arguments for the Infomap algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""

		if args_infomap is None:
			infomap = graph.community_infomap()
		else:
			infomap = graph.community_infomap(**args_infomap)
		return self.from_vertexcluster_tolist(infomap)

	def compute_lab(self, graph: ig.Graph, args_lab: dict) -> List[List[int]]:
		"""
		Compute the Label Propagation community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be clustered
		args_lab : dict
			The arguments for the Label Propagation algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""

		if args_lab is None:
			lab = graph.community_label_propagation()
		else:
			lab = graph.community_label_propagation(**args_lab)
		return self.from_vertexcluster_tolist(lab)

	def compute_louv(self, graph: ig.Graph, args_louv: dict) -> List[List[int]]:
		"""
		Compute the Louvain community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be clustered
		args_louv : dict
			The arguments for the Louvain algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""
		if args_louv is None:
			louv = graph.community_multilevel()
		else:
			louv = graph.community_multilevel(**args_louv)
		return self.from_vertexcluster_tolist(louv)

	def compute_walk(self, graph: ig.Graph, args_walk: dict) -> List[List[int]]:
		"""
		Compute the Walktrap community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be clustered
		args_walk : dict
			The arguments for the Walktrap algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""

		if args_walk is None:
			walk = graph.community_walktrap()
		else:
			walk = graph.community_walktrap(**args_walk)
		# Need to be converted to VertexClustering object
		return self.from_vertexcluster_tolist(walk.as_clustering())

	def compute_eig(self, graph: ig.Graph, args_eig: dict) -> List[List[int]]:
		"""
		Compute the Leading Eigenvector community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be clustered
		args_eig : dict
			The arguments for the Eigenvector algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""

		if args_eig is None:
			eig = graph.community_leading_eigenvector()
		else:
			eig = graph.community_leading_eigenvector(**args_eig)
		return self.from_vertexcluster_tolist(eig)

	def compute_btw(self, graph: ig.Graph, args_btw: dict) -> List[List[int]]:
		"""
		Compute the Edge Betweenness community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be clustered
		args_btw : dict
			The arguments for the Betweenness algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""

		if args_btw is None:
			btw = graph.community_edge_betweenness()
		else:
			btw = graph.community_edge_betweenness(**args_btw)
		return self.from_vertexcluster_tolist(btw.as_clustering())
	
	def compute_spin(self, graph: ig.Graph, args_spin: dict) -> List[List[int]]:
		"""
		Compute the Spin Glass community detection algorithm

		Parameters
		----------
		graph : ig.Graph
			The graph to be clustered
		args_spin : dict
			The arguments for the Spin algorithm

		Returns
		----------
		List[List[int]]
			list of list of vertices in each cluster
		"""

		if args_spin is None:
			spin = graph.community_spinglass()
		else:
			spin = graph.community_spinglass(**args_spin)
		return self.from_vertexcluster_tolist(spin)