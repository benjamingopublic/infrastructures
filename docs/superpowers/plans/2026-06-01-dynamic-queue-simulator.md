# Phase 2a: Dynamic Queue Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vectorised, time-stepped mesoscopic queue simulator for the Copenhagen road network that runs a full rush-hour episode in <<1s, suitable for RL training loops.

**Architecture:** Each edge is a queue with a free-flow traversal time (in 30s steps) and a per-step outflow capacity. Discrete vehicles carry a pre-computed route and a departure step; state is held in flat per-vehicle numpy arrays advanced by vectorised operations each step.

**Tech Stack:** Python 3.10+, numpy, scipy (already installed). No new dependencies.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `src/transport_brain/dynamic_sim.py` | **Create** | `DT`, `SimResult`, `compute_free_flow_routes`, `QueueSim` |
| `tests/test_dynamic_sim.py` | **Create** | All tests for the above |
| `src/transport_brain/sim.py` | **Untouched** | Existing Network + Frank-Wolfe |
| `src/transport_brain/network.py` | **Untouched** | Existing OSM loader |

---

## Setup

Before starting, ensure the package is installed:

```bash
cd transport_brain
pip install -e ".[dev]"
pytest tests/test_assignment.py -v   # existing tests must still pass throughout
```

---

## Task 1: Scaffold + SimResult dataclass

**Files:**
- Create: `src/transport_brain/dynamic_sim.py`
- Create: `tests/test_dynamic_sim.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dynamic_sim.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_dynamic_sim.py -v
```

Expected: `ImportError` — module or names not found.

- [ ] **Step 3: Create the scaffold**

Create `src/transport_brain/dynamic_sim.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_dynamic_sim.py::test_dt_constant tests/test_dynamic_sim.py::test_simresult_fields -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transport_brain/dynamic_sim.py tests/test_dynamic_sim.py
git commit -m "feat: scaffold dynamic_sim.py with DT constant and SimResult dataclass"
```

---

## Task 2: compute_free_flow_routes

**Files:**
- Modify: `src/transport_brain/dynamic_sim.py`
- Modify: `tests/test_dynamic_sim.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dynamic_sim.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_dynamic_sim.py -k "routes" -v
```

Expected: `AttributeError: module has no attribute 'compute_free_flow_routes'`.

- [ ] **Step 3: Implement compute_free_flow_routes**

Append to `src/transport_brain/dynamic_sim.py`:

```python
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
```

- [ ] **Step 4: Run to verify all four route tests pass**

```bash
pytest tests/test_dynamic_sim.py -k "routes" -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transport_brain/dynamic_sim.py tests/test_dynamic_sim.py
git commit -m "feat: add compute_free_flow_routes using free-flow Dijkstra"
```

---

## Task 3: QueueSim.__init__ and reset()

**Files:**
- Modify: `src/transport_brain/dynamic_sim.py`
- Modify: `tests/test_dynamic_sim.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dynamic_sim.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_dynamic_sim.py -k "init or reset" -v
```

Expected: `AttributeError: module has no attribute 'QueueSim'`.

- [ ] **Step 3: Implement QueueSim.__init__, reset(), and _edge_occupancy()**

Append to `src/transport_brain/dynamic_sim.py`:

```python
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
```

- [ ] **Step 4: Run to verify the three init/reset tests pass**

```bash
pytest tests/test_dynamic_sim.py -k "init or reset" -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transport_brain/dynamic_sim.py tests/test_dynamic_sim.py
git commit -m "feat: add QueueSim __init__, reset(), and _edge_occupancy()"
```

---

## Task 4: step() — depart and advance without capacity limits

**Files:**
- Modify: `src/transport_brain/dynamic_sim.py`
- Modify: `tests/test_dynamic_sim.py`

This task implements a simplified `step()` that advances all pending vehicles unconditionally. Capacity enforcement is added in Task 5.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dynamic_sim.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_dynamic_sim.py -k "single_vehicle or departure_offset or two_edge" -v
```

Expected: `AttributeError: 'QueueSim' object has no attribute 'step'`.

- [ ] **Step 3: Implement step() without capacity enforcement**

Append inside the `QueueSim` class in `src/transport_brain/dynamic_sim.py` (before the closing of the class — add after `done` property):

```python
    def step(self, t: int) -> np.ndarray:
        """
        Advance simulation by one DT-second step.
        Returns edge_occupancy[E] — the RL observation.
        Capacity enforcement is not yet applied (added in next iteration).
        """
        # 1. Release vehicles scheduled for this step.
        depart = (self.departure_step == t) & ~self.started
        if depart.any():
            self.current_edge[depart] = self.routes[depart, 0]
            self.exit_step[depart] = t + self.edge_steps[self.current_edge[depart]]
            self.started[depart] = True

        # 2. Advance all vehicles whose edge traversal is complete.
        pending = np.where(self.started & ~self.arrived & (self.exit_step <= t))[0]

        if len(pending) > 0:
            self.route_pos[pending] += 1
            fin_mask = self.route_pos[pending] >= self.route_lengths[pending]
            finished = pending[fin_mask]
            continuing = pending[~fin_mask]

            if len(finished) > 0:
                self.arrived[finished] = True
                self.arrival_step[finished] = t

            if len(continuing) > 0:
                new_edges = self.routes[continuing, self.route_pos[continuing]]
                self.current_edge[continuing] = new_edges
                self.exit_step[continuing] = t + self.edge_steps[new_edges]

        occ = self._edge_occupancy()
        self._max_queue = np.maximum(self._max_queue, occ)
        return occ
```

- [ ] **Step 4: Run to verify the three basic tests pass**

```bash
pytest tests/test_dynamic_sim.py -k "single_vehicle or departure_offset or two_edge" -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transport_brain/dynamic_sim.py tests/test_dynamic_sim.py
git commit -m "feat: implement QueueSim.step() with depart and uncapped advance"
```

---

## Task 5: step() — FIFO capacity enforcement

**Files:**
- Modify: `src/transport_brain/dynamic_sim.py`
- Modify: `tests/test_dynamic_sim.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dynamic_sim.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_dynamic_sim.py::test_capacity_queuing -v
```

Expected: FAIL — without capacity enforcement all three arrive at step 1, so `arrivals == [1, 1, 1]`.

- [ ] **Step 3: Replace step() with the capacity-enforcing version**

In `src/transport_brain/dynamic_sim.py`, replace the entire `step` method with:

```python
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
```

- [ ] **Step 4: Run to verify the capacity test and all prior tests pass**

```bash
pytest tests/test_dynamic_sim.py -v
```

Expected: all tests to this point PASS, including the three from Task 4.

- [ ] **Step 5: Commit**

```bash
git add src/transport_brain/dynamic_sim.py tests/test_dynamic_sim.py
git commit -m "feat: add FIFO per-edge capacity enforcement to QueueSim.step()"
```

---

## Task 6: run(), get_result(), and episode metrics

**Files:**
- Modify: `src/transport_brain/dynamic_sim.py`
- Modify: `tests/test_dynamic_sim.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dynamic_sim.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_dynamic_sim.py -k "mass or delay or queue" -v
```

Expected: `AttributeError: 'QueueSim' object has no attribute 'run'`.

- [ ] **Step 3: Implement get_result() and run()**

Append inside the `QueueSim` class in `src/transport_brain/dynamic_sim.py`:

```python
    def get_result(self) -> SimResult:
        """
        Build SimResult from current state.
        Always call run() before get_result() — run() caps arrival_step
        for non-arrived vehicles so travel times are finite.
        """
        tt = float(np.sum((self.arrival_step - self.departure_step) * DT))
        delay = tt - float(self.free_flow_time_s.sum())
        return SimResult(
            total_travel_time_s=tt,
            total_delay_s=delay,
            max_queue_per_edge=self._max_queue.copy(),
            arrival_steps=self.arrival_step.copy(),
            n_arrived=int(self.arrived.sum()),
        )

    def run(self, n_steps: int = 240) -> SimResult:
        """Run a full episode and return metrics. Resets state first."""
        self.reset()
        for t in range(n_steps):
            self.step(t)
            if self.done:
                break
        # Cap non-arrived vehicles so get_result() travel times are finite.
        not_arrived = ~self.arrived
        if not_arrived.any():
            self.arrival_step[not_arrived] = n_steps
        return self.get_result()
```

- [ ] **Step 4: Run to verify all four metric tests pass**

```bash
pytest tests/test_dynamic_sim.py -k "mass or delay or queue" -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Run the full suite to confirm no regressions**

```bash
pytest -v
```

Expected: all tests PASS, including the original `test_assignment.py` SO≤UE guarantee.

- [ ] **Step 6: Commit**

```bash
git add src/transport_brain/dynamic_sim.py tests/test_dynamic_sim.py
git commit -m "feat: add QueueSim.run() and get_result() with full episode metrics"
```

---

## Task 7: CPH scale performance test

**Files:**
- Modify: `tests/test_dynamic_sim.py`

Validates the `<<1s` target on the real 1539-edge Copenhagen network with 4000 vehicles.

- [ ] **Step 1: Write the performance test**

Append to `tests/test_dynamic_sim.py`:

```python
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
```

- [ ] **Step 2: Run the test with verbose output**

```bash
pytest tests/test_dynamic_sim.py::test_cph_scale_performance -v -s
```

If elapsed > 1s, profile to find the bottleneck:

```python
# Run this in a Python REPL after reproducing the slow path:
import cProfile, pstats
pr = cProfile.Profile()
pr.enable()
result = sim.run(n_steps=240)
pr.disable()
ps = pstats.Stats(pr).sort_stats("cumtime")
ps.print_stats(15)
```

The most likely bottleneck is the `for start, count in zip(g_starts, g_counts)` rank loop. Since it iterates over unique *active edges per step* (at most 1539, typically far fewer), it should be fast. If it isn't, the alternative is to compute ranks via `np.repeat(np.arange(max_count), g_counts)[:len(ps)]` — but try profiling first.

- [ ] **Step 3: Run the full test suite one final time**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_dynamic_sim.py
git commit -m "test: CPH scale performance — 4000 vehicles on real Copenhagen network in <1s"
```

---

## Final Verification

- [ ] Confirm only the two new files were added:

```bash
git log --oneline
git show --stat HEAD~6..HEAD --name-only | grep -v "^$" | sort -u
```

Expected output includes only:
```
src/transport_brain/dynamic_sim.py
tests/test_dynamic_sim.py
```

- [ ] Run the complete test suite one last time:

```bash
pytest -v
```

All tests green. Phase 2a complete.
