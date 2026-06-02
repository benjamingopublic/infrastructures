import numpy as np
import pytest
from transport_brain.sim import Network
from transport_brain.env import CommuteEnv


def make_test_env():
    net = Network(
        edge_from=np.array([0, 1], dtype=np.int32),
        edge_to=np.array([1, 2], dtype=np.int32),
        t0=np.array([30., 60.], dtype=np.float64),
        capacity=np.array([1200., 1200.], dtype=np.float64),
        n_nodes=3,
    )
    node_xy = np.array([[0., 0.], [1., 0.], [2., 0.]])
    attractors = [1, 2]
    return CommuteEnv(
        net=net,
        node_xy=node_xy,
        attractors=attractors,
        n_trips=20,
        n_steps=10,
        max_release=5,
        seed=42,
    )


def test_import():
    env = make_test_env()
    assert env is not None
