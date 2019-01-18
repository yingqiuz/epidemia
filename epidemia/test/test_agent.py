import numpy as np
import pytest
from ..agent import AgentGroup

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
def agents():
    return AgentGroup(ADJ, DIST, WEIGHTS, REGION_SIZE, dt=0.01)


def test_agents(agents):
    assert np.array_equal(agents.adj, ADJ)
    assert np.allclose(agents.dist, ADJ * DIST)
    assert np.allclose(agents.weights, ADJ * WEIGHTS)
    assert np.allclose(agents.region_size, REGION_SIZE)
    assert np.allclose(
        agents.spread_weights,
        ADJ_WEIGHTS / ADJ_WEIGHTS.sum(axis=1)[:, np.newaxis]
    )
    assert np.allclose(
        agents.dist_inv[agents.adj != 0],
        DIST_INVERSE[DIST_INVERSE != 0]
    )
