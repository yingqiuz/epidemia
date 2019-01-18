# -*- coding: utf-8 -*-
"""SIR model"""
import numpy as np
from .agent import AgentGroup
from .utils import (
    growth_process_region,
    clearance_process_region,
    transmission_process_region
)


class SISAgents(AgentGroup):
    def __init__(
            self, adj, dist, weights, region_size, dt=0.01
    ):
        AgentGroup.__init__(
            self, adj, dist, weights, region_size, dt
        )
        self.states = list('SI')

        self.region_to_edge_weights = np.tile(
            self.spread_weights, [len(self.states), 1, 1]
        )

        self.edge_to_region_weights = np.tile(
            self.dist_inv, [len(self.states), 1, 1]
        )

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

    def remove_population(self, region=0, edge=0):
        self.region_population -= region

        # seems okay...
        self.edge_population -= \
            edge * np.tile(self.adj, [len(self.states), 1, 1])

    def set_spread_process(self, prob_stay=1, v=1):
        # prob_stay can be scalar value, or np.ndarray(self.n_regions, )
        # or np.ndarray(len(self.states), self.n_regions)
        # the probability of moving from regions to edges
        prob_stay = np.ones((len(self.states), self.n_regions)) * prob_stay
        self.region_to_edge_weights = np.tile(
            1 - prob_stay[:, :, np.newaxis], [1, 1, self.n_regions]
        ) * self.spread_weights
        # self.region_to_edge_weights[self.diag == 1] = prob_stay.flatten()
        self.region_to_edge_weights += np.tile(
            np.eye(self.n_regions), [len(self.states), 1, 1]
        ) * prob_stay[:, :, np.newaxis]
        # the probability of moving from edges to regions
        # determined by length and v (speed)
        # v can be scalr value, or np.adarray(self.n_regions, self.n_regions)
        # or np.ndarray(len(self.states), self.n_regions, self.n_regions)
        # for now assume all three states have the same speed and prob_stay
        # tile v
        v = np.ones(
            (len(self.states), self.n_regions, self.n_regions)
        ) * v
        self.edge_to_region_weights = v * self.dist_inv

    def spread_step(self):
        region_to_edge = self.region_population[:, :, np.newaxis] * \
                         self.region_to_edge_weights * self.dt
        region_to_edge[self.diag == 1] = 0

        # move from edge to region
        edge_to_region = self.edge_population * \
            self.edge_to_region_weights * self.dt

        self.region_population = self.region_population + (
            edge_to_region.sum(axis=1) - region_to_edge.sum(axis=2)
        )
        self.edge_population = self.edge_population + \
            region_to_edge - edge_to_region

    def region_growth_step(self, growth_rate=0):
        self.region_population[0, :] += growth_process_region(
            growth_rate, self.region_size, self.dt
        )

    def region_recovery_step(self, recovery_rate=0):
        recovered = clearance_process_region(
            self.region_population[1, :], recovery_rate, self.dt
        )
        self.region_population[1, :] -= recovered
        self.region_population[0, :] += recovered

    def region_transmission_step(self, trans_rate):
        infected_population = transmission_process_region(
            self.region_population[0, :],
            self.region_population[1, :],
            trans_rate, self.region_size, self.dt
        )

        self.region_population[0, :] -= infected_population
        self.region_population[1, :] += infected_population
