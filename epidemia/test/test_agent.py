import numpy as np
import pytest
from ..agent import AgentGroup

N_REGIONS = 100
ADJ = np.where(np.random.rand(N_REGIONS, N_REGIONS)>0.5, 1, 0)[0]
DIST = np.random.rand(N_REGIONS, N_REGIONS) * 100
WEIGHTS = np.random.rand(N_REGIONS, N_REGIONS)
REGION_SIZE = region_size = np.random.rand(N_REGIONS)


@pytest.fixture
def agents():
    return AgentGroup(ADJ, DIST, WEIGHTS, REGION_SIZE, dt=0.01)


def test_agents(agents):
    assert np.array_equal(agents.adj, ADJ)
    assert np.allclose(agents.adj_dist, ADJ * DIST)
    assert np.allclose(agents.adj_weights, ADJ * WEIGHTS)
    assert np.allclose(agents.region_size, REGION_SIZE)
    assert np.allclose(
        agents.region_to_edge_weights,
        ADJ * WEIGHTS / (ADJ * WEIGHTS).sum(axis=1)[:, np.newaxis]
    )
    assert np.allclose(
        agents.edge_to_region_weights[agents.adj != 0],
        1 * ADJ / DIST[ADJ != 0]
    )
