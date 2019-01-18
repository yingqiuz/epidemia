# -*- coding: utf-8 -*-
"""
a class to store the agents and do basic operations
"""
import numpy as np


class AgentGroup:
    def __init__(
            self, adj, dist, weights, region_size, dt=0.01
    ):
        self.adj = np.where(np.array(adj) != 0, 1, 0)
        self.n_regions = self.adj.shape[0]

        # make sure the diagnal is zero
        self.adj[np.eye(self.n_regions) == 1] = 0

        self.dist = np.array(dist) * self.adj
        self.weights = np.array(weights) * self.adj

        # binarise dist and weights matrix
        self.dist_inv = np.zeros(self.dist.shape)
        self.dist_inv[self.adj != 0] = 1 / self.dist[self.adj != 0]
        self.spread_weights = self.weights / \
            self.weights.sum(axis=1)[:, np.newaxis]

        # region_size
        self.region_size = np.array(region_size) * \
            np.ones((self.n_regions, ))

        self.growth_rate = 0
        self.clearance_rate = 0
        self.trans_rate = 0
        self.dt = dt
