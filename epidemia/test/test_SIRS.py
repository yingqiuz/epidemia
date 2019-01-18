import numpy as np
import pytest
from ..SIRS import SIRSAgents

N_REGIONS = 100
ADJ = np.where(np.random.rand(N_REGIONS, N_REGIONS) > 0.5, 1, 0)
ADJ[np.eye(N_REGIONS) == 1] = 0
DIST = np.random.rand(N_REGIONS, N_REGIONS) * 100
ADJ_DIST = DIST * ADJ
DIST_INVERSE = np.zeros(DIST.shape)
DIST_INVERSE[ADJ_DIST != 0] = 1 / ADJ_DIST[ADJ_DIST != 0]
WEIGHTS = np.random.rand(N_REGIONS, N_REGIONS)
ADJ_WEIGHTS = WEIGHTS * ADJ
REGION_SIZE = region_size = np.random.rand(N_REGIONS)


@pytest.fixture
def si_agents():
    new_group = SIRSAgents(
        ADJ, DIST, WEIGHTS, REGION_SIZE, dt=0.01
    )
    return new_group


def test_sir(si_agents):
    region = np.random.randint(1, 10)
    edge = np.random.randint(1, 10)
    si_agents.add_population(region=region, edge=edge)
    assert np.array_equal(
        si_agents.region_population,
        np.zeros((3, N_REGIONS)) + region
    )
    assert np.array_equal(
        si_agents.edge_population,
        (np.zeros((3, N_REGIONS, N_REGIONS)) + edge) * ADJ
    )
    # remove agents test
    si_agents.remove_population(region=region, edge=edge)
    assert np.array_equal(
        si_agents.region_population,
        np.zeros((3, N_REGIONS))
    )
    assert np.array_equal(
        si_agents.edge_population,
        np.zeros((3, N_REGIONS, N_REGIONS))
    )
    region = np.random.randint(1, 10, (N_REGIONS, ))
    edge = np.random.randint(1, 10, (N_REGIONS, N_REGIONS))

    si_agents.add_population(region=region, edge=edge)
    assert np.array_equal(
        si_agents.region_population,
        np.zeros((3, N_REGIONS)) + region
    )
    assert np.array_equal(
        si_agents.edge_population,
        (np.zeros((3, N_REGIONS, N_REGIONS)) + edge) * ADJ
    )
    # remove agents test
    si_agents.remove_population(region=region, edge=edge)
    assert np.array_equal(
        si_agents.region_population,
        np.zeros((3, N_REGIONS))
    )
    assert np.array_equal(
        si_agents.edge_population,
        np.zeros((3, N_REGIONS, N_REGIONS))
    )

    prob_stay = np.random.rand(3, N_REGIONS)
    v = np.random.rand(N_REGIONS, N_REGIONS)
    si_agents.set_spread_process(prob_stay=prob_stay, v=v)
    assert np.allclose(
        si_agents.edge_to_region_weights,
        np.tile(
            DIST_INVERSE * v, [3, 1, 1]
        )
    )
    assert np.allclose(
        si_agents.region_to_edge_weights,
        np.tile(
            ADJ_WEIGHTS / ADJ_WEIGHTS.sum(axis=1)[:, np.newaxis],
            [3, 1, 1]
        ) * (1 - prob_stay[:, :, np.newaxis]) +
        np.tile(
            np.eye(N_REGIONS), [3, 1, 1]
        ) * prob_stay[:, :, np.newaxis]
    )

    prob_stay = np.random.rand(N_REGIONS)
    si_agents.set_spread_process(prob_stay=prob_stay, v=1)
    assert np.allclose(
        si_agents.region_to_edge_weights,
        np.tile(
            ADJ_WEIGHTS / ADJ_WEIGHTS.sum(axis=1)[:, np.newaxis] *
            (1 - prob_stay[:, np.newaxis]) +
            np.eye(N_REGIONS) * prob_stay,
            [3, 1, 1]
        )
    )
    assert np.allclose(
        si_agents.edge_to_region_weights,
        np.tile(
            DIST_INVERSE, [3, 1, 1]
        )
    )

    si_agents.add_population(region=100, edge=100)
    si_agents.spread_step()
    region_population = 100 - (1 - prob_stay) * 100 * 0.01 + \
        (100 * 0.01 * DIST_INVERSE).sum(axis=0)
    edge_population = 100 * (
        ADJ_WEIGHTS / ADJ_WEIGHTS.sum(axis=1)[:, np.newaxis] *
        (1 - prob_stay[:, np.newaxis])
    ) * 0.01 - 100 * 0.01 * DIST_INVERSE + 100 * ADJ

    assert np.allclose(
        si_agents.region_population[1],
        region_population
    )
    assert np.allclose(
        si_agents.edge_population[1],
        edge_population
    )

    growth_rate = np.random.rand(N_REGIONS)
    si_agents.region_growth_step(growth_rate=growth_rate)
    assert np.allclose(
        si_agents.region_population[0],
        region_population + growth_rate * REGION_SIZE * 0.01
    )
    clearance_rate = np.random.rand(2, N_REGIONS)
    si_agents.region_clearance_step(clearance_rate)
    assert np.allclose(
        si_agents.region_population[1],
        region_population - region_population * (
            1 - np.exp(-clearance_rate[1] * 0.01)
        )
    )
    assert np.allclose(
        si_agents.region_population[2],
        region_population + region_population * (
            1 - np.exp(-clearance_rate[1] * 0.01)
        ) + (region_population + growth_rate * REGION_SIZE * 0.01) * (
                1 - np.exp(-clearance_rate[0] * 0.01)
        )
    )
    region_s = np.copy(si_agents.region_population[0])
    region_i = np.copy(si_agents.region_population[1])
    trans_rate = np.random.rand(N_REGIONS)
    si_agents.region_transmission_step(trans_rate=trans_rate)
    assert np.allclose(
        si_agents.region_population[1],
        region_i + region_s * (
            1 - np.exp(-trans_rate * region_i * 0.01 / REGION_SIZE)
        )
    )

    region_s = np.copy(si_agents.region_population[0])
    region_r = np.copy(si_agents.region_population[2])
    susceptible_rate = np.random.rand(N_REGIONS)
    si_agents.region_become_susceptible_step(
        susceptible_rate=susceptible_rate
    )
    assert np.allclose(
        si_agents.region_population[0],
        region_s + region_r * (
            1 - np.exp(-susceptible_rate * 0.01)
        )
    )
