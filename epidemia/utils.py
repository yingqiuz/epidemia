# -*- coding: utf-8 -*-
import numpy as np


def growth_process_region(growth_rate, region_size, dt):
    """return newly added population"""
    return growth_rate * region_size * dt


def clearance_process_region(population, clearance_rate, dt):
    """return newly removed population"""
    return population * (1 - np.exp(- clearance_rate * dt))


def transmission_process_region(
        population_s, population_i, trans_rate, region_size, dt
):
    """return newly infected population"""
    return population_s * (
        1 - np.exp(
            - trans_rate * (population_i / region_size) * dt
        )
    )

