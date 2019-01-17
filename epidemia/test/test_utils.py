# -*- coding: utf-8 -*-
import numpy as np
from ..utils import (
    growth_process_region,
    clearance_process_region,
    transmission_process_region,
)


def test_growth_process_region():
    growth_rate = np.random.rand()
    region_size = np.random.rand(100)
    assert np.array_equal(
        growth_process_region(growth_rate, region_size, 0.01),
        growth_rate * region_size * 0.01
    )

    growth_rate = np.random.rand(100)
    region_size = np.random.rand(100)
    assert np.array_equal(
        growth_process_region(growth_rate, region_size, 0.01),
        growth_rate * region_size * 0.01
    )

    growth_rate = np.random.rand(1, 100)
    region_size = np.random.rand(100)
    assert np.array_equal(
        growth_process_region(growth_rate, region_size, 0.01),
        growth_rate * region_size * 0.01
    )


def test_clearance_process_region():
    clearance_rate = np.random.rand(100)
    population = np.random.rand(2, 100)
    assert np.array_equal(
        clearance_process_region(population, clearance_rate, 0.01),
        population * (1 - np.exp(-clearance_rate * 0.01))
    )

    clearance_rate = np.random.rand(2, 100)
    population = np.random.rand(2, 100)
    assert np.array_equal(
        clearance_process_region(population, clearance_rate, 0.01),
        population * (1 - np.exp(-clearance_rate * 0.01))
    )

    clearance_rate = np.random.rand()
    population = np.random.rand(2, 100)
    assert np.array_equal(
        clearance_process_region(population, clearance_rate, 0.01),
        population * (1 - np.exp(-clearance_rate * 0.01))
    )


def test_transmission_process_region():
    trans_rate = np.random.rand()
    s, i, region_size = \
        np.exp(np.random.randn(3, 100)) / \
        (1 + np.exp(np.random.randn(3, 100)))
    assert np.allclose(
        transmission_process_region(
            s, i, trans_rate, region_size, 0.01
        ), s * (1 - np.exp(-i * trans_rate * 0.01 / region_size))
    )

    trans_rate, s, i, region_size = \
        np.exp(np.random.randn(4, 100)) / \
        (1 + np.exp(np.random.randn(4, 100)))
    assert np.allclose(
        transmission_process_region(
            s, i, trans_rate, region_size, 0.01
        ), s * (1 - np.exp(-i * trans_rate * 0.01 / region_size))
    )
