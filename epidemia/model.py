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
