# -*- coding: utf-8 -*-
"""SIR model"""
import numpy as np
from .agent import AgentGroup
from .utils import (
    growth_process_region,
    clearance_process_region,
    transmission_process_region
)


class SIAgents(AgentGroup):
    def __init__(
            self, adj, dist, weights, region_size, dt=0.01
    ):
        AgentGroup.__init__(
            self, adj, dist, weights, region_size, dt
        )
        self.states = list('SI')

        # initialize population in regions
        self.region_population = np.zeros(
            (len(self.states), self.n_regions)
        )

        # initialize the population in paths
        self.edge_population = np.zeros(
            (len(self.states), self.n_regions, self.n_regions)
        )

        # diagonal matrix, useful in computation
        self.diag = np.tile(
            np.eye(self.n_regions), [len(self.states), 1, 1]
        )

    def add_population(self, region=0, edge=0):
        # add population in regions
        # has to be scalar value, or np.ndarray (self.n_regions, )
        # or np.ndarray(len(self.states), self.n_regions)
        self.region_population += region

        # seems okay...
        self.edge_population += \
            edge * np.tile(self.adj, [len(self.states), 1, 1])

    def set_spread_process(self, prob_stay=1, v=1):
        # prob_stay can be scalar value, or np.ndarray(self.n_regions, )
        # or np.ndarray(len(self.states), self.n_regions)
        # the probability of moving from regions to edges
        prob_stay *= np.ones((len(self.states), self.n_regions))
        self.region_to_edge_weights = np.tile(
            1 - prob_stay[:, :, np.newaxis], [1, 1, self.n_regions]
        ) * self.region_to_edge_weights
        self.region_to_edge_weights[self.diag == 1] = prob_stay.flatten()
        # the probability of moving from edges to regions
        # determined by length and v (speed)
        # v can be scalr value, or np.adarray(self.n_regions, self.n_regions)
        # or np.ndarray(len(self.states), self.n_regions, self.n_regions)
        # for now assume all three states have the same speed and prob_stay
        # tile v
        v *= np.ones(
            (len(self.states), self.n_regions, self.n_regions)
        )
        self.edge_to_region_weights *= v

    def spread_step(self):
        region_to_edge = self.region_population[:, :, np.newaxis] * \
                         self.region_to_edge_weights
        region_to_edge = np.array([np.fill_diagonal(x, 0) for x in region_to_edge])

        # move from edge to region
        edge_to_region = self.edge_population * self.edge_to_region_weights

        self.region_population += edge_to_region.sum(axis=1) - region_to_edge.sum(axis=2)
        self.edge_population += region_to_edge - edge_to_region

    def region_growth_step(self, growth_rate=0):
        self.region_population[0, :] += growth_process_region(
            growth_rate, self.region_size, self.dt
        )

    def region_transmission_step(self, trans_rate):
        infected_population = transmission_process_region(
            self.region_population[0, :],
            self.region_population[1, :],
            trans_rate, self.region_size, self.dt
        )
