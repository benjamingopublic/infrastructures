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


def compute_free_flow_routes():
    """Compute free-flow routes. To be implemented."""
    pass
