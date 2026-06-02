# Phase 2b: Gymnasium Environment — Design Spec

**Date:** 2026-06-02
**Status:** Approved
**Scope:** `src/transport_brain/env.py` + `tests/test_env.py` only. No changes to existing files.

---

## Context

Phase 2a produced a fast queue simulator (`QueueSim`, ~0.02s/episode). Phase 2b wraps it in a Gymnasium environment so standard RL libraries (Stable-Baselines3, CleanRL) can train an agent that exploits foreknowledge of trip destinations to smooth departure timing and reduce congestion delay.

The key finding from Phase 1 carries forward: **temporal smoothing dominates spatial rerouting**. The agent's job is to decide *when* vehicles depart, not where they go. Routes are pre-computed at free-flow speed and fixed for the episode.

---

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Action space | Global release-rate (Discrete) | Simplest useful policy; per-trip offsets are Phase 2c |
| Observation | Rich: edge occupancy + zone histogram + queue state | Agent needs spatial demand context, not just queue size |
| Reward | Dense per-step: delay cost + holding cost | Terminal-only reward has credit assignment problems |
| Vehicles released | FIFO by desired departure step | Fair; respects when people want to leave |
| Rendering | `render("human")` live + `render("rgb_array")` for capture | Standard Gymnasium pattern; opt-in so training isn't slowed |

---

## Module Structure

```
src/transport_brain/env.py      # CommuteEnv + record_episode()
tests/test_env.py               # 10 tests
```

No existing files are modified. Requires `pip install -e ".[rl]"` (Gymnasium is already in the `rl` extra).

---

## CommuteEnv Constructor

```python
CommuteEnv(
    net: Network,
    node_xy: np.ndarray,          # shape (N, 2), x/y coords from GraphML
    attractors: list[int],         # attractor node indices for demand generation
    n_trips: int = 4000,
    n_steps: int = 240,            # episode length (240 × 30s = 2 hr)
    max_release: int = 20,         # max vehicles releasable per step
    wait_coeff: float = 1.0,       # inconvenience penalty weight
    max_expected_occ: float = 10.0, # occupancy normalisation divisor
    seed: int | None = None,       # fixed seed → reproducible episodes; None → random
)
```

`node_xy` and `attractors` come from `load_network()` in `network.py`. The env owns a `QueueSim` instance that is reset each episode.

---

## Action Space

`gymnasium.spaces.Discrete(max_release + 1)`

Action `k` releases exactly `min(k, n_queued)` vehicles from the front of the waiting queue. Released vehicles get `departure_step = current_t` in the sim and enter the network immediately.

---

## Episode Structure

**Two departure-time concepts:**
- `desired_departure_step[v]` — when vehicle v *wants* to leave (env-level, immutable per episode, used to determine queue order and holding cost).
- `sim.departure_step[v]` — when vehicle v *actually* departs (set by the env when it releases the vehicle from the queue; starts as a large sentinel so no vehicle auto-departs).

**`reset(seed, options)`:**
1. Generate trips via `make_rush_hour_demand(net, node_xy, attractors, n_trips, seed=episode_seed)`
2. Compute free-flow routes via `compute_free_flow_routes(net, trips)`
3. Assign each trip a `desired_departure_step` drawn uniformly from `[0, n_steps // 2]` (all vehicles want to leave in the first hour — this creates the peak the agent must smooth)
4. Sort trips into a `deque` by `desired_departure_step` (FIFO release order)
5. Build `QueueSim(net, routes, departure_steps=np.full(n_trips, n_steps + 100))` so no vehicle auto-departs; call `sim.reset()`
6. Precompute zone assignments (8 rings by distance from centroid) — done once per reset
7. Return initial observation, info dict

**Releasing vehicles:** when the env releases vehicle `v` at step `t`, it writes `sim.departure_step[v] = t` directly into the sim's numpy array before calling `sim.step(t)`. This is safe because `departure_step` is a plain numpy array attribute — no encapsulation is broken and no changes to `dynamic_sim.py` are required.

**`step(action)`:**
1. Pop `min(action, n_queued)` vehicle indices from front of queue; write `sim.departure_step[v] = t` for each
2. Call `sim.step(t)`
3. Compute reward (see below)
4. Build observation
5. `t += 1`
6. `terminated = (t >= n_steps) or sim.done`
7. Return `obs, reward, terminated, False, info`

`info` dict contains: `n_queued`, `n_in_transit`, `n_arrived`, `step`.

---

## Observation Space

`gymnasium.spaces.Box(low=0, high=np.inf, shape=(1549,), dtype=np.float32)`

| Slice | Dim | Content | Normalisation |
|---|---|---|---|
| `obs[0:1539]` | 1539 | `edge_occupancy` from `sim.step()` | `÷ max_expected_occ` |
| `obs[1539:1547]` | 8 | Origin-zone histogram of queued trips | `÷ n_trips` |
| `obs[1547]` | 1 | `n_queued` | `÷ n_trips` |
| `obs[1548]` | 1 | `t / n_steps` | already [0, 1] |

**Zone assignment:** at `reset()`, each trip's origin node is binned into one of 8 equal-width distance rings from the network centroid, using the `node_xy` coordinates. The mapping is precomputed once per reset and reused for all steps. The histogram counts queued trips per zone at the current step.

Observation bounds are `low=0, high=np.inf` — occupancy can briefly exceed `max_expected_occ` under extreme congestion. SB3 and CleanRL handle unbounded observations correctly.

---

## Reward Function

```python
new_arrived_mask = sim.arrived & ~prev_arrived        # vehicles that arrived this step
delay_cost = sum over new_arrived_mask of:
    (sim.arrival_step[v] - sim.departure_step[v]) * DT - sim.free_flow_time_s[v]
holding_cost = n_queued                               # trips still waiting to be released

reward = -(delay_cost / DT + wait_coeff * holding_cost) / n_trips
```

**Rationale:**
- `delay_cost` captures network congestion — vehicles newly arrived this step pay for edge queuing delay.
- `delay_cost / DT` converts from seconds to "steps of delay" (DT=30s), putting it on the same scale as `holding_cost` (measured in vehicle-steps).
- `holding_cost` prevents the degenerate "release nobody" policy that achieves zero congestion at the cost of zero users served.
- `/ n_trips` normalises reward magnitude so hyperparameters transfer across demand sizes.
- `wait_coeff` (default 1.0) is the key trade-off knob — higher values push earlier release.

Reward is always ≤ 0. Episode return ≈ `−(total_delay_veh_steps + wait_coeff × total_queue_veh_steps) / n_trips`.

---

## Rendering

**`render(mode="human")`:** Updates a persistent matplotlib figure in-place (created lazily on first call):
- Left panel: road network edges coloured by `occupancy / send_capacity[e]` — blue (empty) → yellow → red (saturated). Edges drawn as line segments using precomputed node `x,y` positions.
- Right panel: vertical bar showing `n_queued / n_trips`, step counter, current reward.

**`render(mode="rgb_array")`:** Returns `(H, W, 3)` uint8 array of the same figure. Used for frame capture during recording.

**`record_episode(env, policy_fn, path="episode.mp4") -> SimResult`** (module-level function):
- Runs one full episode, calling `policy_fn(obs) → action` each step
- Captures `render("rgb_array")` each step
- Writes frames to `path` via `matplotlib.animation.FFMpegWriter`
- Returns the final `SimResult` for metric logging alongside the video

Node positions are precomputed at `reset()` from `node_xy` — no OSMnx calls during rendering.

---

## Test Plan

| Test | What it checks |
|---|---|
| `test_gymnasium_api_compliance` | `check_env(env)` passes — spaces, dtypes, reset/step contract |
| `test_reset_returns_valid_obs` | obs shape=(1549,), dtype=float32, no NaN/Inf |
| `test_action_zero_releases_nothing` | action=0 → n_queued unchanged after step |
| `test_action_max_drains_queue` | action=max_release with 5 queued → at most 5 released |
| `test_reward_negative` | reward ≤ 0 at every step |
| `test_holding_cost_discourages_hoarding` | reward(action=0) < reward(action=max) when queue non-empty and network uncongested |
| `test_episode_terminates` | greedy episode (action=max_release always) terminates within n_steps |
| `test_zone_histogram_sums_to_n_queued` | `obs[1539:1547].sum() * n_trips ≈ obs[1547] * n_trips` |
| `test_render_rgb_array_shape` | `render("rgb_array")` returns `(H, W, 3)` uint8 |
| `test_record_episode_creates_file` | `record_episode(env, lambda obs: 10, "tmp.mp4")` creates the file |

---

## Dependencies

Already declared in `pyproject.toml`:
- `gymnasium>=0.29` (under `[rl]` extra)
- `matplotlib>=3.8` (base dep)
- `numpy>=1.26` (base dep)

Install with: `pip install -e ".[rl]"`

---

## Visual Verification Script

`scripts/visualize_episode.py` — not a test, a manual inspection tool:
1. Load Copenhagen network via `load_network("data/cph.graphml")`
2. Build `CommuteEnv` with default params and seed=0
3. Run one episode with a fixed policy (always release 10 vehicles/step)
4. Call `record_episode(env, policy_fn, "outputs/episode.mp4")`
5. Print: total delay (veh-hours), n_arrived, elapsed wall time

Visual correctness is confirmed by watching the output video: edges should be blue at episode start, turn yellow/red during the congestion peak, and return to blue as vehicles arrive.

---

## What This Is Not

- **Not a training script** — `env.py` is the environment only. Training with SB3/CleanRL is a separate script outside this spec.
- **Not a per-trip action space** — the agent controls global release rate. Per-trip offsets are Phase 2c.
- **Not multimodal** — bikes, metro, buses are Phase 3.
