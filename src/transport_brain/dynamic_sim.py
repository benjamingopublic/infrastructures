import warnings
import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from transport_brain.sim import Network

DT = 30  # seconds per time step


@dataclass
class SimResult:
    total_travel_time_s: float
    total_delay_s: float
    max_queue_per_edge: np.ndarray   # int32[E]
    arrival_steps: np.ndarray        # int32[V]; non-arrived vehicles capped at n_steps
    n_arrived: int


class QueueSim:
    """
    Discrete-vehicle mesoscopic queue simulator.

    Vehicles carry pre-computed routes (edge-index sequences).
    Each call to step(t) advances the simulation by one DT-second timestep.
    """

    def __init__(
        self,
        net: Network,
        routes: list[np.ndarray],
        departure_steps: np.ndarray,
    ) -> None:
        V = len(routes)

        # Precomputed per-edge arrays (immutable after init).
        self.edge_steps = np.maximum(1, np.round(net.t0 / DT).astype(np.int32))
        self.send_capacity = np.maximum(
            1, (net.capacity * DT / 3600).astype(np.int32)
        )
        self.n_edges = net.n_edges

        # Per-vehicle immutable data.
        route_lengths = np.array([len(r) for r in routes], dtype=np.int32)
        max_len = int(route_lengths.max()) if V > 0 else 1
        routes_padded = np.full((V, max_len), -1, dtype=np.int32)
        for i, r in enumerate(routes):
            if len(r) > 0:
                routes_padded[i, : len(r)] = r

        free_flow_time_s = np.array(
            [net.t0[r].sum() if len(r) > 0 else 0.0 for r in routes],
            dtype=np.float64,
        )

        self.routes = routes_padded
        self.route_lengths = route_lengths
        self.departure_step = np.asarray(departure_steps, dtype=np.int32)
        self.free_flow_time_s = free_flow_time_s
        self.n_vehicles = V

        # Mutable state — initialised by reset().
        self.route_pos: np.ndarray
        self.current_edge: np.ndarray
        self.exit_step: np.ndarray
        self.started: np.ndarray
        self.arrived: np.ndarray
        self.arrival_step: np.ndarray
        self._max_queue: np.ndarray
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset mutable state. Returns initial edge occupancy (all zeros)."""
        V = self.n_vehicles
        self.route_pos = np.zeros(V, dtype=np.int32)
        self.current_edge = np.full(V, -1, dtype=np.int32)
        self.exit_step = np.full(V, 10**7, dtype=np.int32)
        self.started = np.zeros(V, dtype=bool)
        self.arrived = np.zeros(V, dtype=bool)
        self.arrival_step = np.zeros(V, dtype=np.int32)
        self._max_queue = np.zeros(self.n_edges, dtype=np.int32)
        return self._edge_occupancy()

    def _edge_occupancy(self) -> np.ndarray:
        active = self.started & ~self.arrived
        if not active.any():
            return np.zeros(self.n_edges, dtype=np.int32)
        return np.bincount(
            self.current_edge[active], minlength=self.n_edges
        ).astype(np.int32)

    @property
    def done(self) -> bool:
        return bool(np.all(self.arrived))

    def step(self, t: int) -> np.ndarray:
        """
        Advance simulation by one DT-second step.
        Returns edge_occupancy[E] — the RL observation.

        Capacity is enforced per edge: at most send_capacity[e] vehicles
        may exit edge e per step. Excess vehicles wait one step (FIFO order
        by exit_step — earliest-waiting gets priority).
        """
        # 1. Release vehicles scheduled for this step.
        depart = (self.departure_step == t) & ~self.started
        if depart.any():
            self.current_edge[depart] = self.routes[depart, 0]
            self.exit_step[depart] = t + self.edge_steps[self.current_edge[depart]]
            self.started[depart] = True

        # 2. Find vehicles whose edge traversal is nominally complete.
        pending = np.where(self.started & ~self.arrived & (self.exit_step <= t))[0]

        if len(pending) > 0:
            # 3. FIFO capacity enforcement.
            #    Sort by (current_edge, exit_step) so vehicles are grouped by edge
            #    and, within each group, earliest exit_step (longest-waiting) goes first.
            p_edges = self.current_edge[pending]
            p_exit = self.exit_step[pending]
            order = np.lexsort((p_exit, p_edges))  # primary: edge; secondary: exit_step
            ps = pending[order]
            es = p_edges[order]

            # Compute each vehicle's rank within its edge group.
            unique_edges, g_starts, g_counts = np.unique(
                es, return_index=True, return_counts=True
            )
            rank = np.empty(len(ps), dtype=np.int32)
            for start, count in zip(g_starts, g_counts):
                rank[start : start + count] = np.arange(count, dtype=np.int32)

            cap = self.send_capacity[es]
            adv_mask = rank < cap

            can_advance = ps[adv_mask]
            blocked = ps[~adv_mask]

            # 4. Advance vehicles that cleared capacity.
            if len(can_advance) > 0:
                self.route_pos[can_advance] += 1
                fin_mask = (
                    self.route_pos[can_advance] >= self.route_lengths[can_advance]
                )
                finished = can_advance[fin_mask]
                continuing = can_advance[~fin_mask]

                if len(finished) > 0:
                    self.arrived[finished] = True
                    self.arrival_step[finished] = t

                if len(continuing) > 0:
                    new_edges = self.routes[continuing, self.route_pos[continuing]]
                    self.current_edge[continuing] = new_edges
                    self.exit_step[continuing] = t + self.edge_steps[new_edges]

            # 5. Delay blocked vehicles — they retry next step.
            if len(blocked) > 0:
                self.exit_step[blocked] = t + 1

        occ = self._edge_occupancy()
        self._max_queue = np.maximum(self._max_queue, occ)
        return occ


def compute_free_flow_routes(
    net: Network, trips: list[tuple[int, int]]
) -> list[np.ndarray]:
    """
    Compute free-flow shortest-path routes for each trip.

    Returns a list of int32 arrays of ordered edge indices.
    Unreachable trips return an empty array and emit a UserWarning.
    """
    mat = csr_matrix(
        (net.t0, (net.edge_from, net.edge_to)),
        shape=(net.n_nodes, net.n_nodes),
    )
    by_origin: dict[int, list[tuple[int, int]]] = {}
    for trip_idx, (o, d) in enumerate(trips):
        by_origin.setdefault(o, []).append((trip_idx, d))

    origins = list(by_origin.keys())
    _, pred = sp_dijkstra(mat, directed=True, indices=origins, return_predecessors=True)

    routes: list[np.ndarray] = [None] * len(trips)  # type: ignore[list-item]
    for row, o in enumerate(origins):
        preds = pred[row]
        for trip_idx, d in by_origin[o]:
            edges: list[int] = []
            node = d
            while node != o and preds[node] >= 0:
                p = int(preds[node])
                edges.append(net._edge_index[(p, node)])
                node = p
            if node != o:
                warnings.warn(
                    f"Trip {trip_idx}: no path from {o} to {d} — vehicle skipped",
                    UserWarning,
                    stacklevel=2,
                )
                routes[trip_idx] = np.array([], dtype=np.int32)
            else:
                routes[trip_idx] = np.array(edges[::-1], dtype=np.int32)
    return routes
