from __future__ import annotations

import io
import numpy as np
import gymnasium as gym
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
        obs_mode: str = "full",
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
        assert obs_mode in ("full", "compact"), "obs_mode must be 'full' or 'compact'"
        self.obs_mode = obs_mode

        self.action_space = gym.spaces.Discrete(max_release + 1)
        obs_dim = net.n_edges + 10 if obs_mode == "full" else 10
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
        self._released: np.ndarray | None = None
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

        # 2. Assign desired departure steps (uniform over full episode).
        # Demand rate ~17/step; max_release=20 so the agent can keep up.
        self._desired_dep = rng.integers(0, self.n_steps, size=self.n_trips).astype(np.int32)

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
        self._released = np.zeros(self.n_trips, dtype=bool)
        self._prev_arrived = np.zeros(self.n_trips, dtype=bool)
        self._last_occ = np.zeros(self.net.n_edges, dtype=np.int32)

        return self._make_obs(), self._make_info()

    def _make_obs(self) -> np.ndarray:
        if self.obs_mode == "compact":
            obs = np.empty(10, dtype=np.float32)
            obs[:8] = self._zone_counts / self.n_trips
            obs[8] = len(self._queue) / self.n_trips
            obs[9] = self._t / self.n_steps
            return obs
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
            self._released[v] = True
            origin_node = self._trips[v][0]
            self._zone_counts[self._node_zone[origin_node]] -= 1

        self._last_occ = self._sim.step(self._t)

        # Cost 1: vehicles still in queue past their desired departure time.
        # Computed after release so the action immediately reduces this cost.
        n_late = int(np.sum((self._desired_dep <= self._t) & ~self._released))
        late_cost = n_late / self.n_trips

        # Cost 2: travel delay for vehicles that arrived this step.
        new_arrived_mask = self._sim.arrived & ~self._prev_arrived
        travel_delay_s = 0.0
        if new_arrived_mask.any():
            idx = np.where(new_arrived_mask)[0]
            travel_steps = self._sim.arrival_step[idx] - self._sim.departure_step[idx]
            travel_delay_s = float(
                np.sum(travel_steps * DT - self._sim.free_flow_time_s[idx])
            )

        reward = -(late_cost + travel_delay_s / (DT * self.n_trips))

        self._prev_arrived = self._sim.arrived.copy()
        self._t += 1

        terminated = (self._t >= self.n_steps) or bool(self._sim.done and len(self._queue) == 0)
        return self._make_obs(), float(reward), terminated, False, self._make_info()

    def render(self) -> np.ndarray | None:
        mode = self.render_mode
        if self._sim is None:
            return None

        if self._fig is None:
            self._fig, self._axes = plt.subplots(1, 2, figsize=(12, 5))
            self._fig.tight_layout(pad=2.0)

        ax_net, ax_bar = self._axes
        ax_net.clear()
        ax_bar.clear()

        # Left panel: road network coloured by load (occupancy / send_capacity).
        occ = self._last_occ if self._last_occ is not None else np.zeros(self.net.n_edges)
        send_cap = np.maximum(1, (self.net.capacity * DT / 3600).astype(np.int32))
        load = np.clip(occ / send_cap, 0, 1)

        segs = []
        colors = []
        for e in range(self.net.n_edges):
            u = int(self.net.edge_from[e])
            v = int(self.net.edge_to[e])
            segs.append([self.node_xy[u], self.node_xy[v]])
            r, g, b = float(load[e]), float(1.0 - load[e]), 0.0
            colors.append((r, g, b, 0.8))

        lc = LineCollection(segs, colors=colors, linewidths=1.0)
        ax_net.add_collection(lc)
        ax_net.autoscale()
        ax_net.set_aspect("equal")
        ax_net.set_title(f"Step {self._t}/{self.n_steps}")
        ax_net.axis("off")

        # Right panel: queue bar + stats.
        frac = len(self._queue) / max(self.n_trips, 1)
        ax_bar.bar([0], [frac], color="steelblue")
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([])
        ax_bar.set_ylabel("Queue fraction")
        ax_bar.set_title(f"Queued: {len(self._queue)}")

        self._fig.canvas.draw()

        if mode == "rgb_array":
            try:
                from PIL import Image
                with io.BytesIO() as buf:
                    self._fig.savefig(buf, format="png", dpi=72)
                    buf.seek(0)
                    img = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
            except ImportError:
                w, h = self._fig.canvas.get_width_height()
                buf_raw = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = buf_raw.reshape(h, w, 3)
            return img
        elif mode == "human":
            plt.pause(0.001)
        return None

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None


def record_episode(
    env: CommuteEnv,
    policy_fn,
    path: str = "episode.mp4",
) -> SimResult:
    import matplotlib.animation as animation

    obs, _ = env.reset()
    frames = []
    terminated = truncated = False

    while not (terminated or truncated):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action = policy_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    # Capture final frame after last step.
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    # Build SimResult from final sim state — only count vehicles that actually departed.
    # Vehicles still in queue have departure_step = NEVER sentinel (n_steps+100), which
    # would produce negative travel times if naively included.
    sim = env._sim
    started_not_arrived = sim.started & ~sim.arrived
    if started_not_arrived.any():
        sim.arrival_step[started_not_arrived] = env.n_steps

    mask = sim.started
    if mask.any():
        tt = float(np.sum((sim.arrival_step[mask] - sim.departure_step[mask]) * DT))
        delay = tt - float(sim.free_flow_time_s[mask].sum())
    else:
        tt = delay = 0.0
    result = SimResult(
        total_travel_time_s=tt,
        total_delay_s=delay,
        max_queue_per_edge=sim._max_queue.copy(),
        arrival_steps=sim.arrival_step.copy(),
        n_arrived=int(sim.arrived.sum()),
    )

    # Write video.
    if frames:
        h, w, _ = frames[0].shape
        fig, ax = plt.subplots(figsize=(w / 72, h / 72), dpi=72)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        im = ax.imshow(frames[0])

        def update(frame_data):
            im.set_data(frame_data)
            return [im]

        writer = animation.FFMpegWriter(fps=10)
        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=100, blit=True
        )
        anim.save(path, writer=writer)
        plt.close(fig)

    env.close()
    return result
