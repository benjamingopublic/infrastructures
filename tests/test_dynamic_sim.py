import time
import numpy as np
import pytest
from transport_brain.sim import Network
from transport_brain.dynamic_sim import DT, SimResult, QueueSim, compute_free_flow_routes


def tiny_net():
    """Single edge: node 0 -> node 1, 30s free-flow, capacity 120 veh/h."""
    return Network(
        edge_from=[0],
        edge_to=[1],
        t0=[30.0],
        capacity=[120.0],
        n_nodes=2,
    )


def two_edge_net():
    """Two edges in sequence: 0->1->2. First=30s, second=60s. High capacity."""
    return Network(
        edge_from=[0, 1],
        edge_to=[1, 2],
        t0=[30.0, 60.0],
        capacity=[1200.0, 1200.0],
        n_nodes=3,
    )


def test_dt_constant():
    assert DT == 30


def test_simresult_fields():
    r = SimResult(
        total_travel_time_s=90.0,
        total_delay_s=0.0,
        max_queue_per_edge=np.array([1], dtype=np.int32),
        arrival_steps=np.array([3], dtype=np.int32),
        n_arrived=1,
    )
    assert r.total_travel_time_s == 90.0
    assert r.n_arrived == 1
