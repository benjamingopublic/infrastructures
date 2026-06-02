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
    import gymnasium as gym
    env = make_test_env()
    assert isinstance(env, gym.Env)


def test_spaces_defined():
    import gymnasium as gym
    env = make_test_env()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 6  # max_release=5, so 0..5
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (12,)  # 2 edges + 10 scalars
    assert env.observation_space.dtype == np.float32


def test_zone_precompute():
    env = make_test_env()
    # _node_zone must be set and contain values 0-7
    assert hasattr(env, '_node_zone')
    assert env._node_zone.shape == (3,)  # 3 nodes in tiny net
    assert env._node_zone.min() >= 0
    assert env._node_zone.max() <= 7
