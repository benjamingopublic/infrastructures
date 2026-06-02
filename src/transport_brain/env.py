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
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self._init_seed
        rng = np.random.default_rng(episode_seed)

        # 1. Generate trips and routes.
        demand = make_rush_hour_demand(self.net, self.node_xy, self.attractors, self.n_trips, seed=episode_seed)
        trips: list[tuple[int, int]] = []
        for (o, d), qty in demand.items():
            trips.extend([(o, d)] * int(qty))
        trips = trips[: self.n_trips]
        while len(trips) < self.n_trips:
            trips.append(trips[rng.integers(len(trips))])

        routes = compute_free_flow_routes(self.net, trips)

        # 2. Assign desired departure steps (uniform in first half of episode).
        self._desired_dep = rng.integers(0, self.n_steps // 2 + 1, size=self.n_trips).astype(np.int32)

        # 3. Build FIFO queue sorted by desired departure step.
        order = np.argsort(self._desired_dep, kind="stable")
        self._queue = deque(int(v) for v in order)

        # 4. Build zone counts for queued vehicles.
        self._zone_counts = np.zeros(8, dtype=np.int32)
        for v in self._queue:
            origin_node = trips[v][0]
            self._zone_counts[self._node_zone[origin_node]] += 1

        # 5. Create QueueSim with NEVER sentinel (no vehicle auto-departs).
        departure_steps = np.full(self.n_trips, self.n_steps + 100, dtype=np.int32)
        self._sim = QueueSim(self.net, routes, departure_steps)

        # 6. Store trips for zone lookup on release.
        self._trips = trips

        self._t = 0
        self._prev_arrived = np.zeros(self.n_trips, dtype=bool)
        self._last_occ = np.zeros(self.net.n_edges, dtype=np.int32)

        return self._make_obs(), self._make_info()

    def _make_obs(self) -> np.ndarray:
        obs = np.empty(self.net.n_edges + 10, dtype=np.float32)
        obs[:self.net.n_edges] = self._last_occ / self.max_expected_occ
        obs[self.net.n_edges: self.net.n_edges + 8] = self._zone_counts / self.n_trips
        obs[self.net.n_edges + 8] = len(self._queue) / self.n_trips
        obs[self.net.n_edges + 9] = self._t / self.n_steps
        return obs

    def _make_info(self) -> dict:
        n_arrived = int(self._sim.arrived.sum()) if self._sim is not None else 0
        n_in_transit = int((self._sim.started & ~self._sim.arrived).sum()) if self._sim is not None else 0
        return {
            "n_queued": len(self._queue),
            "n_in_transit": n_in_transit,
            "n_arrived": n_arrived,
            "step": self._t,
        }

    def step(self, action: int):
        assert self._sim is not None, "call reset() before step()"
        n_release = min(int(action), len(self._queue))
        for _ in range(n_release):
            v = self._queue.popleft()
            self._sim.departure_step[v] = self._t
            origin_node = self._trips[v][0]
            self._zone_counts[self._node_zone[origin_node]] -= 1

        self._last_occ = self._sim.step(self._t)

        # Reward: delay cost for newly-arrived vehicles + holding cost.
        new_arrived_mask = self._sim.arrived & ~self._prev_arrived
        delay_cost = 0.0
        if new_arrived_mask.any():
            idx = np.where(new_arrived_mask)[0]
            travel_steps = self._sim.arrival_step[idx] - self._sim.departure_step[idx]
            delay_cost = float(
                np.sum(travel_steps * DT - self._sim.free_flow_time_s[idx])
            )
        holding_cost = len(self._queue)
        reward = -(delay_cost / DT + self.wait_coeff * holding_cost) / self.n_trips

        self._prev_arrived = self._sim.arrived.copy()
        self._t += 1

        terminated = (self._t >= self.n_steps) or bool(self._sim.done and len(self._queue) == 0)
        return self._make_obs(), float(reward), terminated, False, self._make_info()

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


def record_episode(env: CommuteEnv, policy_fn, path: str = "episode.mp4") -> SimResult:
    raise NotImplementedError
