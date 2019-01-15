# -*- coding: utf-8 -*-
"""
basic classes and infrastructure
"""
import numpy as np
import tqdm
from .agent import AgentGroup
from .utils import (
    growth_process_region, clearance_process_region, transmission_process_region, spread_process
)


class AgentBasedModel:
    def __init__(self, adj, dist, weights, region_size, dt=0.01, method='SIR'):
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

        # region_size
        self.region_size = np.array(region_size)

        # define model
        self.method = method

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

    def add_agent_group(
            self, name, init_region_s=0, init_edge_s=0, init_region_i=0,
            init_edge_i=0, init_region_r=0, init_edge_r=0
    ):
        NewAgentGroup = AgentGroup(
            name, region_s=init_region_s, region_i=init_region_i,
            region_r=init_region_r, edge_s=init_edge_s, edge_i=init_edge_i,
            edge_r=init_edge_r
        )

        self.agent_groups[name] = NewAgentGroup

    def add_growth_process(self, name, growth_rate=0.5):
        self.growth_process[name] = growth_rate

    def add_clearance_process(self, name, clearance_rate=0.5):
        self.clearance_process[name] = clearance_rate

    def add_transmission_process(self, name, trans_rate=1):
        self.trans_process[name] = trans_rate

    def add_spread_process(self, name, weights=None, dist=None, v=1):
        if weights is None:
            self.spread_process[name] = self.weights

    def growth(self, agent_groups=None, growth_t=100000000):
        if agent_groups is None:
            agent_groups = [agent_group for agent_group in self.agent_groups]

        # begin the growth prcoess
        for t in tqdm(range(growth_t)):
            for agent_group_name in agent_groups:
                self.agent_groups[agent_group_name].region_s =
