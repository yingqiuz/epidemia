# -*- coding: utf-8 -*-
"""
a class to store the agents and do basic operations
"""
import numpy as np
from .utils import (growth_process_region,
                    clearance_process_region,
                    transmission_process_region)


class AgentGroup:
    def __init__(
            self, adj, dist, weights, region_size,
            dt=0.01, method='SIR', keep=None
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

        # create the list of states
        # self.states = sorted(set(method), key=lambda x: list(method).index(x))
        self.states = list(method)

        # initialize the population in regions
        self.region_population = np.zeros((len(self.states), self.adj.shape[0]))

        # initialize the population in paths
        self.edge_population = np.zeros(
            (len(self.states), self.adj.shape[0], self.adj.shape[1])
        )

        self.diag = np.tile(
            np.eye(self.n_regions), [len(self.states), 1, 1]
        )

    def add_population(self, region=0, edge=0):
        """
        add population to the model
        :param region: int, (N_region, ) or (N_region, k) numpy_array
        :param edge:
        :return:
        """
        self.region_population += region

        # seems okay...
        self.edge_population += \
            edge * np.tile(self.adj, [len(self.states), 1, 1])

    def add_growth_process(self, growth_rate):
        self.growth_rate = growth_rate

        # or a func obj

    def add_clearance_process(self, clearance_rate, clearance_type='SI'):
        if clearance_type == 'I':
            self.clearance_rate = clearance_rate * np.array(
                [np.zeros((1, self.n_regions)), np.ones((1, self.n_regions))]
            )
        elif clearance_type == 'S':
            self.clearance_rate = clearance_rate * np.array(
                [np.ones((1, self.n_regions)), np.zeros((1, self.n_regions))]
            )
        elif clearance_type == 'SI':
            self.clearance_rate = clearance_rate * np.ones((2, self.n_regions))
        else:
            raise ValueError(
                "{} is not valid. "
                "Clearance_type should be one of 'S', 'I', or 'SI'."
                .format(clearance_rate)
            )

        # or a func obj

    def add_trans_process(self, trans_rate):
        self.trans_rate = trans_rate

        # or a func obj

    def add_spread_process(self, prob_stay=1, v=1):
        # the probability of moving from regions to edges
        prob_stay *= np.ones((len(self.states), self.n_regions))
        self.region_to_edge_weights = np.tile(
            1 - prob_stay[:, :, np.newaxis], [1, 1, self.n_regions]
        ) * self.region_to_edge_weights
        self.region_to_edge_weights[self.diag == 1] = prob_stay.flatten()
        # the probability of moving from edges to regions
        # determined by length and v (speed)
        # for now assume all three states have the same speed and prob_stay
        # tile v
        v *= np.ones(
            (len(self.states), self.n_regions, self.n_regions)
        )
        self.edge_to_region_weights *= v

    def region_growth_step(self):
        self.region_population[0, :] += growth_process_region(
            self.growth_rate, self.region_size, self.dt
        )

    def region_clearance_step(self):
        self.region_population[:2, :] -= clearance_process_region(
            self.region_population[:2, :], self.clearance_rate, self.dt
        )

    def region_transmission_step(self):
        infected_population = transmission_process_region(
            self.region_population[0], self.region_population[1],
            self.region_size, self.trans_rate, self.dt)
        self.region_population[0] -= infected_population
        self.region_population[1] += infected_population

    def spread_process_step(self):
        region_to_edge = self.region_population[:, :, np.newaxis] * \
                         self.region_to_edge_weights
        region_to_edge = np.array([np.fill_diagonal(x, 0) for x in region_to_edge])

        # move from edge to region
        edge_to_region = self.edge_population * self.edge_to_region_weights

        self.region_population += edge_to_region.sum(axis=1) - region_to_edge.sum(axis=2)
        self.edge_population += region_to_edge - edge_to_region
