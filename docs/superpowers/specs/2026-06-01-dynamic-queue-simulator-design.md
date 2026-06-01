# Phase 2a: Fast Dynamic Queue Simulator — Design Spec

**Date:** 2026-06-01  
**Status:** Approved  
**Scope:** `dynamic_sim.py` + `test_dynamic_sim.py` only. No changes to existing files.

---

## Context

Phase 1 produced a static Frank-Wolfe assignment solver (~40s per solve). Phase 2 needs a dynamic simulator that can sit inside an RL training loop — target <<1s per episode. The key finding from Phase 1 is that temporal smoothing dominates spatial rerouting, so the simulator's primary job is to faithfully model queue buildup and dissipation as vehicles depart at different times.

---

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vehicle representation | Discrete individual vehicles | Needed for per-trip inconvenience penalty in RL reward |
| Time step | 30 seconds | 240 steps/2hr episode; fine enough to resolve short Copenhagen edges |
| Routing | Passed in externally; helper computes free-flow shortest paths by default | Keeps sim agnostic; enables Phase 2b to swap in dynamic rerouting |
| Vectorisation strategy | Flat per-vehicle numpy arrays, no Python loops over edges/vehicles | Pure numpy, <<1s target |

---

## Module Structure

```
src/transport_brain/dynamic_sim.py   # all new code
tests/test_dynamic_sim.py            # all new tests
```

No existing files are modified. `dynamic_sim.py` imports `Network` from `sim.py`. No ML dependencies.

---

## Constants

```python
DT = 30  # seconds per time step
```

---

## Precomputed Per-Edge Arrays

Computed once at `QueueSim.__init__` from the `Network`:

```python
edge_steps[E]      = max(1, round(t0[e] / DT))          # free-flow traversal time in steps
send_capacity[E]   = max(1, int(capacity[e] * DT / 3600))  # max vehicles exiting edge per step
```

`capacity[e]` is in veh/hour. Minimum of 1 for both to avoid deadlocks on very short or low-capacity edges.

---

## Per-Vehicle State Arrays

Length V (one entry per trip). Immutable at init, mutable during simulation:

```python
# Immutable (set at init, never changed)
routes           # int32, shape (V, max_route_len), padded with -1
route_lengths    # int32[V] — number of edges in each route
departure_step   # int32[V] — step when vehicle enters its first edge
free_flow_time_s # float64[V] — sum of t0[e] for each edge in route (for delay calc)

# Mutable (reset between episodes)
route_pos        # int32[V] — index of current edge in routes[v]
current_edge     # int32[V] — edge index; -1 = not yet started
exit_step        # int32[V] — step when vehicle will finish current edge
started          # bool[V]
arrived          # bool[V]
arrival_step     # int32[V] — filled when vehicle arrives
```

Routes are stored as a padded 2D array so that `routes[v, route_pos[v]]` is a vectorised fancy-index operation.

---

## Step Algorithm

`step(t: int) -> np.ndarray` — advances simulation by one 30s step, returns `edge_occupancy[E]`.

**1. Depart** — release vehicles scheduled for this step:
```
mask = (departure_step == t)
current_edge[mask] = routes[mask, 0]
exit_step[mask] = t + edge_steps[current_edge[mask]]
started[mask] = True
```

**2. Find pending** — vehicles whose edge traversal is nominally complete:
```
pending = where(started & ~arrived & (exit_step <= t))
```

**3. FIFO capacity enforcement:**
- Sort `pending` by `exit_step` ascending (earliest = waited longest = highest priority)
- Compute each vehicle's rank within its current-edge group using `np.unique` + `np.arange` per group
- `can_advance = rank < send_capacity[current_edge[pending]]`
- Cost: O(unique active edges this step), not O(V) or O(E)

**4. Advance vehicles that cleared capacity:**
```
route_pos[can_advance] += 1
finished   = can_advance & (route_pos >= route_lengths)
arrived[finished] = True;  arrival_step[finished] = t
continuing = can_advance & ~finished
current_edge[continuing] = routes[continuing, route_pos[continuing]]
exit_step[continuing]    = t + edge_steps[current_edge[continuing]]
```

**5. Delay blocked vehicles — retry next step:**
```
exit_step[pending[~can_advance]] = t + 1
```

**Edge occupancy** (returned as RL observation):
```
np.bincount(current_edge[started & ~arrived], minlength=E)
```

---

## Route Computation Helper

```python
def compute_free_flow_routes(net: Network, trips: list[tuple[int, int]]) -> list[np.ndarray]:
```

- Groups trips by origin, runs one Dijkstra (on `t0`) per unique origin
- Walks predecessors back to build an ordered list of edge indices per trip
- Returns list of int32 arrays (one per trip)
- Unreachable trips return empty array; `QueueSim` marks them arrived immediately with zero travel time and logs a warning

**Free-flow lower bound** is computed here as `sum(t0[e] for e in route)` per vehicle — before any congestion. Stored in `free_flow_time_s[V]`. This is the denominator for delay, consistent with the Phase 1 convention (never compute the lower bound under congestion).

---

## Public API

```python
# Construct once per network + trip set
sim = QueueSim(net, routes, departure_steps)

# RL usage — caller drives the loop
obs = sim.reset()            # edge_occupancy[E], all zeros
for t in range(n_steps):
    obs = sim.step(t)        # edge_occupancy[E]
    if sim.done: break
result = sim.get_result()    # SimResult

# Script usage — run to completion internally
result = sim.run(n_steps=240)   # calls reset() + full step loop
```

`reset()` reinitialises all mutable arrays from the immutable route/departure data. The same `QueueSim` object can run thousands of episodes without reallocation.

---

## SimResult

```python
@dataclass
class SimResult:
    total_travel_time_s: float      # veh-seconds, sum of (arrival_step - departure_step) * DT
    total_delay_s: float            # veh-seconds, total_travel_time_s - sum(free_flow_time_s)
    max_queue_per_edge: np.ndarray  # int32[E], worst occupancy seen per edge across all steps
    arrival_steps: np.ndarray       # int32[V]; vehicles not arrived by n_steps get arrival_step = n_steps
    n_arrived: int                  # vehicles that reached destination within the episode window
```

Vehicles still in-transit at `n_steps` are counted as arriving at `n_steps` (capped travel time). `n_arrived < n_vehicles` signals the episode window was too short or congestion is severe.

---

## Test Plan

| Test | What it checks |
|---|---|
| `test_single_vehicle_arrives` | 1 vehicle, 1 edge: arrives exactly at `departure_step + edge_steps[0]` |
| `test_capacity_queuing` | 3 vehicles depart simultaneously on edge with `send_capacity=1`: arrive at t, t+1, t+2 |
| `test_two_edge_route` | 1 vehicle, 2 edges: total steps = `edge_steps[0] + edge_steps[1]` |
| `test_mass_conservation` | Full episode: `sum(arrived) == n_vehicles` at end |
| `test_delay_ordering` | Peak-hour departure pattern produces more delay than spread departures |
| `test_free_flow_lower_bound` | With zero congestion (1 vehicle), `total_delay_s ≈ 0` |
| `test_cph_scale_performance` | 4000 vehicles on real Copenhagen network: `run()` completes in < 1s |

---

## What This Is Not

- **Not a rerouting simulator** — routes are fixed at episode start. Dynamic rerouting is Phase 2b.
- **Not a replacement for Frank-Wolfe** — the static solver remains the oracle/baseline. This is the fast training environment.
- **Not a microscopic sim** — no individual vehicle kinematics, no turn penalties, no signal timing. Mesoscopic queue dynamics only.
