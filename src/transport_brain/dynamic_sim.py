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
    """Dynamic queue simulator. To be implemented."""
    pass


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
