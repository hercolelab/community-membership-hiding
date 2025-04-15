"""Module for the agent class"""
from src.methods.drl_agent.a2c.a2c import ActorCritic
from src.graph_environment.env import GraphEnvironment
from src.utils.utils import DRL_agentHyps, FilePaths, Utils

from tqdm import trange
from collections import namedtuple
from typing import List, Tuple
from torch_geometric.data import Data
from torch.nn import functional as F

from itertools import product
import networkx as nx
import numpy as np
import random
import torch
import json
import gc


class Agent:
    def __init__(
            self,
            env: GraphEnvironment,
            state_dim: int = DRL_agentHyps.EMBEDDING_DIM.value,
            hidden_size_1: int = DRL_agentHyps.HIDDEN_SIZE_1.value,
            hidden_size_2: int = DRL_agentHyps.HIDDEN_SIZE_2.value,
            lr: List[float] = DRL_agentHyps.LR.value,
            gamma: List[float] = DRL_agentHyps.GAMMA.value,
            lambda_metrics: List[float] = DRL_agentHyps.LAMBDA.value,
            alpha_metrics: List[float] = DRL_agentHyps.ALPHA.value,
            epsilon_probs: float = DRL_agentHyps.EPSILON.value,
            weight_decay: float = DRL_agentHyps.WEIGHT_DECAY.value,
            dropout: float = DRL_agentHyps.DROPOUT.value,
            eps: float = DRL_agentHyps.EPS_CLIP.value,
            best_reward: float = DRL_agentHyps.BEST_REWARD.value)-> None:
        """
        Initialize the agent.

        Parameters
        ----------
        env : GraphEnvironment
            Environment to train the agent on
        state_dim : int
            Dimensions of the state, i.e. length of the feature vector
        hidden_size_1 : int
            First A2C hidden layer size
        hidden_size_2 : int
            Second A2C hidden layer size
        action_dim : int
            Dimensions of the action (it is set to 1, to return a tensor N*1)
        lr : List[float]
            List of Learning rate, each element of the list is a learning rate
        gamma : List[float]
            List of gamma parameter, each element of the list is a gamma
        lambda_metrics : List[float]
            List of lambda parameter, each element of the list is a lambda used
            to balance the reward and the penalty
        alpha_metrics : List[float]
            List of alpha parameter, each element of the list is a alpha used
            to balance the two penalties
        eps : List[float]
            Value for clipping the loss function, each element of the list is a
            clipping value
        best_reward : float, optional
            Best reward, by default 0.8
        """
        # ° ----- Environment ----- ° #
        self.env = env

        # ° ----- A2C ----- ° #
        self.state_dim = state_dim # self.env.graph.number_of_nodes()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.action_dim = self.env.graph.number_of_nodes()
        self.dropout = dropout
        self.policy = ActorCritic(
            state_dim=self.state_dim,
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            action_dim=self.action_dim,
            dropout=self.dropout,
        )
        # Set device
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        # Move model to device
        self.policy.to(self.device)

        # ° ----- Hyperparameters ----- ° #
        # A2C hyperparameters
        self.lr_list = lr
        self.gamma_list = gamma
        self.eps = eps
        self.best_reward = best_reward
        self.epsilon_probs = epsilon_probs
        # Environment hyperparameters
        self.lambda_metrics = lambda_metrics
        self.alpha_metrics = alpha_metrics
        self.weight_decay = weight_decay
        # Hyperparameters to be set during grid search
        self.lr = None
        self.gamma = None
        self.alpha_metric = None
        self.epsilon_prob = None
        self.optimizers = dict()

        # ------ Training ------ #
        # TO DO

        # ------ Evaluation ------ #
        # List of actions performed during the evaluation
        self.action_list = {"remove": [], "add": []}

    ############################################################################
    #                       PRE-TRAINING/TESTING                               #
    ############################################################################
    def reset_hyperparams(
            self,
            lr: float,
            gamma: float,
            lambda_metric: float,
            alpha_metric: float,
            epsilon_prob: float,
            test: bool = False) -> None:
        """
        Reset hyperparameters
        
        Parameters
        ----------
        lr : float
            Learning rate
        gamma : float
            Discount factor
        lambda_metric : float
            Lambda parameter used to balance the reward and the penalty
        alpha_metric : float
            Alpha parameter used to balance the two penalties
        epsilon_prob : float
            Probability of changing the target node and the target community
        test : bool, optional
            Print hyperparameters during training, by default False
        """
        # Set A2C hyperparameters
        self.lr = lr
        self.gamma = gamma
        if epsilon_prob < 0 or epsilon_prob > 100:
            raise ValueError("Epsilon must be between 0 and 100")
        self.epsilon_prob = epsilon_prob
        # Set environment hyperparameters
        self.env.lambda_metric = lambda_metric
        self.env.alpha_metric = alpha_metric
        # Print hyperparameters if we are not testing
        if not test:
            self.print_hyperparams()
        # Clear logs, except for the training episodes
        for key in self.log_dict.keys():
            if key != 'train_episodes':
                self.log_dict[key] = list()
        # Clear action list
        self.saved_actions = []
        self.rewards = []
        self.episode_rewards = []
        # Clear state
        self.obs = None
        self.episode_reward = 0
        self.episode_entropy = 0
        self.best_reward = DRL_agentHyps.BEST_REWARD.value
        self.done = False
        self.goal = False
        self.step = 0
        self.optimizers = dict()
        
        # Reset the weights of the policy network
        del self.policy
        self.policy = ActorCritic(
            state_dim=self.state_dim,
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            action_dim=self.action_dim,
            dropout=self.dropout,
        )
        # Set device
        self.policy.to(self.device)

    def configure_optimizers(self) -> None:
        """
        Configure optimizers
        
        Returns
        -------
        optimizers : dict
            Dictionary of optimizers
        """
        actor_params = list(self.policy.actor.parameters())
        critic_params = list(self.policy.critic.parameters())
        self.optimizers['a_optimizer'] = torch.optim.Adam(
            actor_params, lr=self.lr, weight_decay=self.weight_decay)
        self.optimizers['c_optimizer'] = torch.optim.Adam(
            critic_params, lr=self.lr, weight_decay=self.weight_decay)



    def rewiring(self, test=False) -> None:
        """
        Rewiring step, select action and take step in environment.
        
        Parameters
        ----------
        test : bool, optional
            If True, print rewiring action, by default False
        """
        # Select action: return a list of the probabilities of each action
        action_rl, entropy = self.select_action(self.obs)
        torch.cuda.empty_cache()
        # Save rewiring action if we are testing
        if test:
            edge = (self.env.node_target, action_rl)
            if edge in self.env.possible_actions["ADD"]:
                if not self.env.graph.has_edge(*edge):
                    # print("* ADD", edge)
                    self.action_list["ADD"].append(edge)
            elif edge in self.env.possible_actions["REMOVE"]:
                if self.env.graph.has_edge(*edge):
                    # print("* REMOVE", edge)
                    self.action_list["REMOVE"].append(edge)
            
            # Take the action in the environment without computing the reward
            self.obs, self.done = self.env.act(action_rl)
            return
        
        # Take action in environment, compute reward and check if the goal is
        # reached
        self.obs, reward, self.done, self.goal = self.env.step(action_rl)
        # Update episode reward and entropy
        self.episode_entropy += entropy
        self.episode_reward += reward
        # Store the transition in memory, used for the training step
        self.rewards.append(reward)
        # Used for logging
        self.episode_rewards.append(reward)
        self.step += 1

    def select_action(self, state: nx.Graph) -> int:
        """
        Select action, given a state, using the policy network.
        
        Parameters
        ----------
        state : nx.Graph
            Current state of the environment
        
        Returns
        -------
        action: int
            Integer representing a node in the graph, it will be the destination
            node of the rewiring action
        """
        concentration, value = self.policy(state)
        dist = torch.distributions.Categorical(concentration)
        entropy = dist.entropy().mean()
        action = dist.sample()
        self.saved_actions.append(
            self.SavedAction(dist.log_prob(action), value))
        return int(action.item()), entropy

    ############################################################################
    #                               TEST                                       #
    ############################################################################
    def test(
            self,
            lr: float,
            gamma: float,
            lambda_metric: float,
            alpha_metric: float,
            epsilon_prob: float,
            model_path: str,
            graph_reset=True) -> nx.Graph:
        """Hide a given node from a given community"""
        # Set hyperparameters to select the correct folder
        self.reset_hyperparams(lr, gamma, lambda_metric, alpha_metric, epsilon_prob, True)
        # Load best performing model
        self.load_checkpoint(path=model_path)
        # Set model in evaluation mode
        self.policy.eval()
        self.obs = self.env.reset(graph_reset)
        # Rewiring the graph until the target node is isolated from the
        # target community
        while not self.done and self.step < self.env.max_steps:
            self.rewiring(test=True)
        return self.obs

    ############################################################################
    #                            CHECKPOINTING                                 #
    ############################################################################
    def get_path(self) -> str:
        """
        Return the path of the folder where to save the plots and the logs
        
        Returns
        -------
        file_path : str
            Path to the correct folder
        """
        file_path = FilePaths.LOG_DIR.value + \
            f"{self.env.env_name}/{self.env.detection_alg}/" +\
            f"eps-{self.epsilon_prob}/" +\
            f"lr-{self.lr}/gamma-{self.gamma}/" +\
            f"lambda-{self.env.lambda_metric}/alpha-{self.env.alpha_metric}"
        return file_path

    def load_checkpoint(self, path=None):
        """Load checkpoint"""
        if path is None:
            log_dir = self.get_path()
            path = f'{log_dir}/model.pth'
        
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model'])
        for key, _ in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    

    
