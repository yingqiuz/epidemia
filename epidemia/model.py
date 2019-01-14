# -*- coding: utf-8 -*-
"""
basic classes and infrastructure
"""
import numpy as np
DEFAULT_PROCESS = ['growth', 'transmission', 'recover', 'move']
DEFAULT_STATES = ['S', 'I', 'R']


class AgentBasedModel:
    def __init__(
            self,
            adj=None,
            dist=None,
            weight=None,
            growth_t=None,
            spread_t=10000,
            process=None,
            states=None,
            method='population'
    ):
        """

        :param adj:
        :param dist:
        :param weight:
        :param growth_t: int
        :param spread_t: int
        :param process: list, optional
            the list of all the processes at each step
        :param states: list, optional
            the states that the agents can take
        :param method: str, optional
            'population', metapopulation model
            'agent', agent-based model
        """
        self.adj = np.where(np.array(adj) !=0, 1, 0)
        self.dist = np.array(dist)
        self.weight = np.array(weight)

        # binarized distance and weight matrix
        self.adj_dist = self.dist * self.adj
        self.adj_weight = self.dist * self.adj

        # set default time
        self.growth_t = growth_t
        self.spread_t = spread_t

        # set up default process
        if process is None:
            self.process = DEFAULT_PROCESS
        if states is None:
            self.states = DEFAULT_STATES


