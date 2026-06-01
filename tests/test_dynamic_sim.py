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


def test_capacity_queuing():
    """
    3 vehicles depart simultaneously on an edge with send_capacity=1/step.
    Without capacity enforcement all three arrive at step 1.
    With enforcement they arrive at steps 1, 2, 3.
    """
    net = tiny_net()   # send_capacity = max(1, int(120*30/3600)) = 1
    routes = compute_free_flow_routes(net, [(0, 1)] * 3)
    sim = QueueSim(net, routes, np.array([0, 0, 0]))

    for t in range(10):
        sim.step(t)

    assert sim.arrived.all()
    arrivals = sorted(sim.arrival_step.tolist())
    assert arrivals == [1, 2, 3]


def test_mass_conservation():
    """Every vehicle must arrive by end of a sufficiently long episode."""
    net = tiny_net()   # send_capacity=1/step
    n = 10
    routes = compute_free_flow_routes(net, [(0, 1)] * n)
    # 10 vehicles all departing step 0; capacity=1/step needs at least 10 steps
    sim = QueueSim(net, routes, np.zeros(n, dtype=np.int32))
    result = sim.run(n_steps=50)
    assert result.n_arrived == n


def test_zero_delay_no_congestion():
    """Single vehicle on high-capacity edge: delay must be zero."""
    net = Network(
        edge_from=[0], edge_to=[1],
        t0=[30.0], capacity=[100_000.0],
        n_nodes=2,
    )
    routes = compute_free_flow_routes(net, [(0, 1)])
    sim = QueueSim(net, routes, np.array([0]))
    result = sim.run(n_steps=10)
    # travel_time = 1 step * 30s = 30s; free_flow_time = 30s; delay = 0
    assert result.total_delay_s == pytest.approx(0.0, abs=1.0)
    assert result.n_arrived == 1


def test_delay_increases_with_congestion():
    """Peak departures produce more delay than spread departures."""
    net = tiny_net()   # send_capacity=1/step creates congestion
    n = 6
    routes = compute_free_flow_routes(net, [(0, 1)] * n)

    peak_sim = QueueSim(net, routes, np.zeros(n, dtype=np.int32))
    peak = peak_sim.run(n_steps=30)

    spread_sim = QueueSim(net, routes, np.arange(n, dtype=np.int32))
    spread = spread_sim.run(n_steps=30)

    assert peak.total_delay_s > spread.total_delay_s


def test_simresult_max_queue():
    """max_queue_per_edge must record the worst congestion seen."""
    net = tiny_net()
    routes = compute_free_flow_routes(net, [(0, 1)] * 5)
    sim = QueueSim(net, routes, np.zeros(5, dtype=np.int32))
    result = sim.run(n_steps=20)
    assert result.max_queue_per_edge[0] >= 1


def test_cph_scale_performance():
    """
    4000 vehicles on the real Copenhagen network must complete in under 1 second.
    Skipped automatically if data/cph.graphml is not present.
    """
    import os
    from transport_brain.network import load_network, make_rush_hour_demand

    graphml = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "cph.graphml")
    )
    if not os.path.exists(graphml):
        pytest.skip("data/cph.graphml not found — skipping scale test")

    net, node_ids, node_xy, attractors = load_network(graphml)
    demand = make_rush_hour_demand(net, node_xy, attractors, n_trips=4000, seed=42)
    trips = list(demand.keys())

    # Spread departures across 1-hour window to avoid extreme queues.
    rng = np.random.default_rng(42)
    departure_steps = rng.integers(0, 120, size=len(trips), dtype=np.int32)

    routes = compute_free_flow_routes(net, trips)
    sim = QueueSim(net, routes, departure_steps)

    t0 = time.perf_counter()
    result = sim.run(n_steps=240)
    elapsed = time.perf_counter() - t0

    print(f"\nCPH scale: {len(trips)} vehicles, {elapsed:.3f}s")
    print(f"  arrived={result.n_arrived}/{len(trips)}")
    print(f"  total_delay={result.total_delay_s / 3600:.1f} veh-hours")

    assert elapsed < 1.0, f"Episode took {elapsed:.2f}s — target <1s"
    assert result.n_arrived > len(trips) * 0.95, "Too many vehicles stuck"
