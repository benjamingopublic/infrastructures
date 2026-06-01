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


def test_routes_single_edge():
    net = tiny_net()
    routes = compute_free_flow_routes(net, [(0, 1)])
    assert len(routes) == 1
    assert routes[0].tolist() == [0]


def test_routes_two_edges():
    net = two_edge_net()
    routes = compute_free_flow_routes(net, [(0, 2)])
    assert len(routes) == 1
    assert routes[0].tolist() == [0, 1]


def test_routes_unreachable_returns_empty():
    net = tiny_net()  # only edge is 0->1; no path 1->0
    with pytest.warns(UserWarning, match="no path"):
        routes = compute_free_flow_routes(net, [(1, 0)])
    assert len(routes[0]) == 0


def test_routes_multiple_trips_grouped_by_origin():
    net = two_edge_net()
    routes = compute_free_flow_routes(net, [(0, 1), (0, 2)])
    assert routes[0].tolist() == [0]
    assert routes[1].tolist() == [0, 1]


def test_init_precomputed_edge_arrays():
    net = tiny_net()  # t0=[30.0], capacity=[120.0]
    routes = compute_free_flow_routes(net, [(0, 1)])
    sim = QueueSim(net, routes, np.array([0]))
    # edge_steps = max(1, round(30/30)) = 1
    assert sim.edge_steps[0] == 1
    # send_capacity = max(1, int(120 * 30 / 3600)) = max(1, 1) = 1
    assert sim.send_capacity[0] == 1


def test_init_routes_padded():
    net = two_edge_net()
    routes = compute_free_flow_routes(net, [(0, 2), (0, 1)])
    sim = QueueSim(net, routes, np.array([0, 0]))
    assert sim.routes.shape == (2, 2)       # 2 vehicles, max_route_len=2
    assert sim.route_lengths[0] == 2        # trip 0->2 uses both edges
    assert sim.route_lengths[1] == 1        # trip 0->1 uses only edge 0


def test_reset_zeroes_mutable_state():
    net = tiny_net()
    routes = compute_free_flow_routes(net, [(0, 1)])
    sim = QueueSim(net, routes, np.array([0]))
    sim.reset()
    assert not sim.started[0]
    assert not sim.arrived[0]
    assert sim.current_edge[0] == -1


def test_single_vehicle_arrives():
    """One vehicle, one 30s edge: arrives at departure_step + edge_steps = 0 + 1 = 1."""
    net = tiny_net()
    routes = compute_free_flow_routes(net, [(0, 1)])
    sim = QueueSim(net, routes, np.array([0]))

    for t in range(10):
        sim.step(t)

    assert sim.arrived[0]
    assert sim.arrival_step[0] == 1


def test_departure_offset():
    """Vehicle departing at step 5 on a 1-step edge arrives at step 6."""
    net = tiny_net()
    routes = compute_free_flow_routes(net, [(0, 1)])
    sim = QueueSim(net, routes, np.array([5]))

    for t in range(10):
        sim.step(t)

    assert sim.arrived[0]
    assert sim.arrival_step[0] == 6


def test_two_edge_route():
    """
    Single vehicle, two edges: edge 0 = 30s (1 step), edge 1 = 60s (2 steps).
    Departs step 0, exits edge 0 at step 1, exits edge 1 at step 3.
    """
    net = two_edge_net()
    routes = compute_free_flow_routes(net, [(0, 2)])
    sim = QueueSim(net, routes, np.array([0]))

    for t in range(10):
        sim.step(t)

    assert sim.arrived[0]
    assert sim.arrival_step[0] == 3
