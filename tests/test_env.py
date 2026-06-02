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


def test_reset_returns_valid_obs():
    env = make_test_env()
    obs, info = env.reset()
    assert obs.shape == (12,)
    assert obs.dtype == np.float32
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()
    assert obs.min() >= 0.0


def test_reset_info_keys():
    env = make_test_env()
    _, info = env.reset()
    for key in ("n_queued", "n_in_transit", "n_arrived", "step"):
        assert key in info, f"missing key: {key}"


def test_zone_histogram_sums_to_n_queued():
    env = make_test_env()
    obs, info = env.reset()
    n_queued = info["n_queued"]
    zone_sum = obs[env.net.n_edges:env.net.n_edges + 8].sum() * env.n_trips
    assert abs(zone_sum - n_queued) < 1.0  # float32 rounding tolerance


def test_action_zero_releases_nothing():
    env = make_test_env()
    env.reset()
    n_before = len(env._queue)
    env.step(0)
    assert len(env._queue) == n_before


def test_action_max_drains_queue():
    env = make_test_env()
    env.reset()
    n_before = len(env._queue)
    obs, reward, terminated, truncated, info = env.step(env.max_release)
    released = n_before - len(env._queue)
    assert released == min(env.max_release, n_before)


def test_reward_negative():
    env = make_test_env()
    env.reset()
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.max_release)
        assert reward <= 0.0, f"reward={reward} should be <= 0"
        if terminated:
            break


def test_holding_cost_discourages_hoarding():
    """action=0 (hoard) should give worse reward than action=max when queue non-empty."""
    env_hold = make_test_env()
    env_hold.reset()
    if len(env_hold._queue) == 0:
        pytest.skip("queue empty at t=0 for this seed")
    _, reward_hold, _, _, _ = env_hold.step(0)

    env_release = make_test_env()
    env_release.reset()
    _, reward_release, _, _, _ = env_release.step(env_release.max_release)

    assert reward_hold < reward_release, (
        f"hoarding reward {reward_hold} should be worse (more negative) than releasing {reward_release}"
    )


def test_episode_terminates():
    env = make_test_env()
    env.reset()
    for step_num in range(env.n_steps + 10):
        _, _, terminated, truncated, _ = env.step(env.max_release)
        if terminated or truncated:
            break
    assert terminated or truncated, "episode did not terminate"
    assert step_num < env.n_steps + 5
