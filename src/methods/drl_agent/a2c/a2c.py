"""Module for the ActorCritic class"""
from src.methods.drl_agent.a2c.actor import ActorNetwork
from src.methods.drl_agent.a2c.critic import CriticNetwork
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from torch.nn import functional as F
from torch import nn
from collections import defaultdict
from typing import Any, Tuple, Optional, Union, List, Literal, Dict
import igraph as ig
import torch
from torch import Tensor
from torch_geometric.data import Data


class ActorCritic(nn.Module):
    """ActorCritic Network"""

    def __init__(
        self, 
        state_dim: int, 
        hidden_size_1: int, 
        hidden_size_2: int, 
        action_dim: int,
        dropout: float):
        super(ActorCritic, self).__init__()
        self.actor = ActorNetwork(
            state_dim=state_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            action_dim=action_dim,
            dropout=dropout,
        )
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            dropout=dropout,
        )
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        

    def forward(self, graph: ig.Graph, jitter=1e-20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, computes action and value

        Parameters
        ----------
        graph : ig.Graph
            Graph state
        jitter : float, optional
            Jitter value, by default 1e-20

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of concentration and value
        """
        # Convert graph to torch_geometric.data.Data
        state = from_igraph(graph).to(self.device)

        # Actor
        probs = self.actor(state)        
        # Use softplus to ensure concentration is positive, then add jitter to 
        # ensure numerical stability
        concentration = F.softplus(probs).reshape(-1) + jitter

        # Critic
        value = self.critic(state)
        return concentration, value




def from_igraph(
    G,
    group_node_attrs: Optional[Union[List[str], Literal['all']]] = None,
    group_edge_attrs: Optional[Union[List[str], Literal['all']]] = None,
):
    r"""Converts an :class:`igraph.Graph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (igraph.Graph): An igraph graph.
        group_node_attrs (List[str] or "all", optional): The node attributes to be
            concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or "all", optional): The edge attributes to be
            concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.

    Examples:
        >>> # Create a simple igraph graph with 4 nodes and 3 edges.
        >>> from igraph import Graph
        >>> G = Graph()
        >>> G.add_vertices(4)
        >>> G.add_edges([(0, 1), (1, 2), (2, 3)])
        >>> # Assign a numeric attribute to nodes and edges.
        >>> G.vs["feat"] = [0.1, 0.2, 0.3, 0.4]
        >>> G.es["weight"] = [1, 2, 3]
        >>> data = from_igraph(G, group_node_attrs="all", group_edge_attrs="all")
        >>> print(data)
    """

    G = G.as_directed(mode="mutual") if not G.is_directed() else G

    mapping = {v.index: i for i, v in enumerate(G.vs)}
    edge_index = torch.empty((2, G.ecount()), dtype = torch.long)
    for i, e in enumerate(G.es):
        edge_index[0, i] = mapping[e.source]
        edge_index[1, i] = mapping[e.target]

    data_dict: Dict[str, Any] = defaultdict(list)
    data_dict['edge_index'] = edge_index

    node_attrs: List[str] = []
    if G.vcount() > 0:
        node_attrs = list(G.vs[0].attributes().keys())

    edge_attrs: List[str] = []
    if G.ecount() > 0:
        edge_attrs = list(G.es[0].attributes().keys())

    if group_node_attrs is not None and not isinstance(group_node_attrs, list):
        group_node_attrs = node_attrs
    
    if group_edge_attrs is not None and not isinstance(group_edge_attrs, list):
        group_edge_attrs = edge_attrs

    for i, v in enumerate(G.vs):
        feat_dict = v.attributes()
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data_dict[str(key)].append(value)
    
    for i, e in enumerate(G.es):
        feat_dict = e.attributes()
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data_dict[str(key)].append(value)

    for key, value in dict(G.attributes()).items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data_dict[str(key)] = value

    for key, value in data_dict.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data_dict[key] = torch.stack(value, dim=0)
        else:
            try:
                data_dict[key] = torch.as_tensor(value)
            except Exception:
                pass

    data = Data.from_dict(data_dict)

    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)
    
    if data.x is None and data.pos is None:
        data.num_nodes = G.vcount()

    return data
