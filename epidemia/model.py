# -*- coding: utf-8 -*-
"""
basic classes and infrastructure
"""
import numpy as np
import tqdm
from .SI import SIAgents
from .SIR import SIRAgents
from .SIRS import SIRSAgents
from .SIS import SISAgents


class AgentBasedModel:
    def __init__(self, adj, dist, weights, region_size, dt=0.01):
        """

        :param adj:
        :param dist:
        :param weights:
        :param region_size:
        :param method:
        """
        self.adj = np.where(np.array(adj) != 0, 1, 0)
        self.dist = np.array(dist)
        self.weights = np.array(weights)

        # binarized dist and weights matrix
        self.adj_dist = self.dist * self.adj
        self.adj_weights = self.dist * self.adj
        self.spread_weights = self.adj_weights / self.adj_weights.sum(axis=1)

        # region_size
        self.region_size = np.array(region_size)

        # initialize agent groups
        self.agent_groups = dict()

        # initialize growth process
        self.growth_process = dict()

        # initialize clearance process
        self.clearance_process = dict()

        # initialize transmission process
        self.trans_process = dict()

        # initialize spreading process
        self.spread_process = dict()

        self.dt = dt

    def add_agents(self, group_name, epidemic_type):
        if epidemic_type == 'SI':
            self.agent_groups[group_name] = SIAgents(
                self.adj, self.dist, self.weights,
                self.region_size, dt=0.01
            )
        if epidemic_type == 'SIR':
            self.agent_groups[group_name] = SIRAgents(
                self.adj, self.dist, self.weights,
                self.region_size, dt=0.01
            )
        if epidemic_type == 'SIS':
            self.agent_groups[group_name] = SISAgents(
                self.adj, self.dist, self.weights,
                self.region_size, dt=0.01
            )
        if epidemic_type == 'SIRS':
            self.agent_groups[group_name] = SIRSAgents(
                self.adj, self.dist, self.weights,
                self.region_size, dt=0.01
            )