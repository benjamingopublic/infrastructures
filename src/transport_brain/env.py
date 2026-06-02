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
        self.render_mode = render_mode

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
