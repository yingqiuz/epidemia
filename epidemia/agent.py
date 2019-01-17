# -*- coding: utf-8 -*-
"""
a class to store the agents and do basic operations
"""
import numpy as np
from .utils import (
    growth_process_region, clearance_process_region, transmission_process_region
)


class AgentGroup:
    def __init__(
            self, adj, dist, weights, region_size, dt=0.01
    ):
        self.adj = np.where(np.array(adj) != 0, 1, 0)
        self.dist = np.array(dist)
        self.weights = np.array(weights)
        self.n_regions = self.adj.shape[0]

        # binarise dist and weights matrix
        self.adj_dist = self.dist * self.adj
        self.adj_weights = self.dist * self.adj

        # region_size
        self.region_size = np.array(region_size)

        self.region_to_edge_weights = \
            self.adj_weights / self.adj_weights.sum(axis=1)

        self.edge_to_region_weights = np.zeros(self.dist.shape)
        self.edge_to_region_weights[self.adj_dist != 0] = \
            1 / self.adj_dist[self.adj_dist != 0]

        self.growth_rate = 0
        self.clearance_rate = 0
        self.trans_rate = 0
        self.dt = dt

