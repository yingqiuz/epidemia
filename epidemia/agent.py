# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from .utils import (growth_process_region,
                    clearance_process_region,
                    transmission_process_region)


class AgentGroup:
    def __init__(
            self,
            adj, dist, weights, region_size, group_name,
            region_s=0, region_i=0, region_r=0,
            edge_s=0, edge_i=0, edge_r=0, dt=0.01,
            method='SIR'
    ):
        self.adj = np.where(np.array(adj) != 0, 1, 0)
        self.dist = np.array(dist)
        self.weights = np.array(weights)

        # binarized dist and weights matrix
        self.adj_dist = self.dist * self.adj
        self.adj_weights = self.dist * self.adj

        # region_size
        self.region_size = np.array(region_size)

        self.region_to_edge_weights = self.adj_weights / self.adj_weights.sum(axis=1)
        self.edge_to_region_weights = np.zeros(self.dist.shape)
        self.edge_to_region_weights[self.adj_dist != 0] = 1 / self.adj_dist

        self.growth_rate = 0
        self.recover_rate = 0
        self.trans_rate = 0
        self.dt = dt

        # initialize the population in regions
        self.region_population = np.zeros((len(method), self.adj.shape[0]))

        # initialize the population in paths
        self.edge_population = np.zeros((len(method), self.adj.shape[0], self.adj.shape[1]))

        # create the list of states
        self.states = sorted(set(method), key=lambda x: list(method).index(x))

    def add_population(self, region=0, edge=0):
        """
        add population to the model
        :param region: int, (N_region, ) or (N_region, k) numpy_array
        :param edge:
        :return:
        """
        self.region_population += region
        self.edge_population += edge

    def add_growth_process(self, growth_rate):
        self.growth_rate = growth_rate

    def add_recover_process(self, recover_rate):
        self.recover_rate = recover_rate

    def add_trans_process(self, trans_rate):
        self.trans_rate = trans_rate

    def add_spread_process(self, prob_stay=1, v=1):
        if isinstance(prob_stay, (list, np.ndarray)):
            self.region_to_edge_weights = np.array([
                np.fill_diagonal(self.region_to_edge_weights * (1 - x), x)
                for x in prob_stay
            ])
        else:
            self.region_to_edge_weights = np.fill_diagonal(
                self.region_to_edge_weights * (1 - prob_stay), prob_stay
            )

        if isinstance(v, (list, np.ndarray)):
            self.edge_to_region_weights = np.array([
                self.edge_to_region_weights * x
                for x in v
            ])
        else:
            self.edge_to_region_weights *= v

    def region_growth_step(self):
        self.region_population += growth_process_region(self.growth_rate,
                                                        self.region_size,
                                                        self.dt)

    def region_recovery_step(self):
        self.region_population -= clearance_process_region(self.region_population,
                                                           self.recover_rate,
                                                           self.dt)

    def region_transmission_step(self):
        infected_population = transmission_process_region(self.region_population[0],
                                                          self.region_population[1],
                                                          self.region_size,
                                                          self.trans_rate, self.dt)
        self.region_population[0] -= infected_population
        self.region_population[1] += infected_population

    def spread_process_step(self):
        region_to_edge = self.region_population[:, :, np.newaxis] * self.region_to_edge_weights
        region_to_edge = np.array([np.fill_diagonal(x, 0) for x in region_to_edge])

        # move from edge to region
        edge_to_region = self.edge_population * self.edge_to_region_weights

        self.region_population += edge_to_region.sum(axis=1) - region_to_edge.sum(axis=2)
        self.edge_population += region_to_edge - edge_to_region

    


