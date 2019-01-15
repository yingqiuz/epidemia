# -*- coding: utf-8 -*-
"""

"""
import numpy as np


class AgentGroup:
    def __init__(
            self,
            adj, dist, weights, region_size, group_name,
            region_s=0, region_i=0, region_r=0,
            edge_s=0, edge_i=0, edge_r=0
    ):
        self.adj = np.where(np.array(adj) != 0, 1, 0)
        self.dist = np.array(dist)
        self.weights = np.array(weights)

        # binarized dist and weights matrix
        self.adj_dist = self.dist * self.adj
        self.adj_weights = self.dist * self.adj

        # region_size
        self.region_size = np.array(region_size)


