from __future__ import annotations

import numpy as np
import gymnasium as gym
from collections import deque

from transport_brain.sim import Network
from transport_brain.dynamic_sim import QueueSim, SimResult, compute_free_flow_routes, DT
from transport_brain.network import make_rush_hour_demand


class CommuteEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        net: Network,
        node_xy: np.ndarray,
        attractors: list[int],
        n_trips: int = 4000,
        n_steps: int = 240,
        max_release: int = 20,
        wait_coeff: float = 1.0,
        max_expected_occ: float = 10.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.node_xy = np.asarray(node_xy, dtype=np.float64)
        self.attractors = list(attractors)
        self.n_trips = int(n_trips)
        self.n_steps = int(n_steps)
        self.max_release = int(max_release)
        self.wait_coeff = float(wait_coeff)
        self.max_expected_occ = float(max_expected_occ)
        self._init_seed = seed
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(max_release + 1)
        obs_dim = net.n_edges + 10
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Precompute 8-zone ring assignment (distance from centroid).
        centroid = self.node_xy.mean(axis=0)
        dists = np.linalg.norm(self.node_xy - centroid, axis=1)
        max_dist = dists.max()
        self._node_zone = np.floor(
            dists / (max_dist + 1e-9) * 8
        ).astype(np.int32).clip(0, 7)

        # Mutable state (set by reset)
        self._sim: QueueSim | None = None
        self._queue: deque[int] = deque()
        self._desired_dep: np.ndarray | None = None
        self._t: int = 0
        self._prev_arrived: np.ndarray | None = None
        self._last_occ: np.ndarray | None = None
        self._zone_counts: np.ndarray = np.zeros(8, dtype=np.int32)
        self._trips: list | None = None
        self._fig = None
        self._axes = None

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


def record_episode(env: CommuteEnv, policy_fn, path: str = "episode.mp4") -> SimResult:
    raise NotImplementedError
