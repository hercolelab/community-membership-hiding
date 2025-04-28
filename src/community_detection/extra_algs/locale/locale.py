from sdp_clustering import leiden_locale
from scipy.io import mmread
import igraph as ig
from cdlib import NodeClustering
from collections import defaultdict
import os
import numpy as np

def ig_leiden_locale(graph: ig.Graph, name: str) -> None:
    """
    Run the leiden-LOCALE algorithm on the given ig.Graph.
    We transform the graph to a matrixMarket format into temp_graph.mtx file, and then apply the algorithm.

    Parameters
    ----------
    graph : ig.Graph
        The input graph.
    
    name : str
        The name of the graph, used for saving the file.
    
    Returns
    -------
    NodeClustering
        The detected communities as a NodeClustering object.
    """

    # Save the graph to a matrixMarket format file
    dir_path = f"src/community_detection/extra_algs/locale/temp_graphs/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = f"temp_graph_{name}.mtx"
    filepath = os.path.join(dir_path, file_name)

    n = graph.vcount()
    m = graph.ecount()
    edges = graph.get_edgelist()
    with open(filepath, 'w') as f:
        # Write the header
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"{n} {n} {m}\n")
        # Write each edge (1-based indexing)
        for u, v in edges:
            f.write(f"{u+1} {v+1}\n")  
    
    # Read the graph from the matrixMarket file
    mtx_graph = mmread(filepath)
    # Run the leiden-LOCALE algorithm
    memberships = leiden_locale(mtx_graph)
    # Convert the memberships to a NodeClustering object
    coms_to_node = defaultdict(list)
    for n, c in enumerate(memberships):
        coms_to_node[c].append(n)
    coms = [list(c) for c in coms_to_node.values()]
    final_coms = NodeClustering(
        coms,
        graph,
        method_name="SCD",
        overlap=False,
    )

    # Remove the temporary file
    os.remove(filepath)
    return final_coms
