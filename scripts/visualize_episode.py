#!/usr/bin/env python3
"""Manual visual inspection: run one episode and write outputs/episode.mp4."""
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transport_brain.network import load_network
from transport_brain.env import CommuteEnv, record_episode

os.makedirs("outputs", exist_ok=True)

net, node_xy, attractors = load_network("data/cph.graphml")
env = CommuteEnv(
    net=net,
    node_xy=node_xy,
    attractors=attractors,
    n_trips=4000,
    n_steps=240,
    max_release=20,
    seed=0,
    render_mode="rgb_array",
)

t0 = time.perf_counter()
result = record_episode(env, lambda obs: 10, path="outputs/episode.mp4")
elapsed = time.perf_counter() - t0

print(f"n_arrived:    {result.n_arrived} / 4000")
print(f"total delay:  {result.total_delay_s / 3600:.2f} veh-hours")
print(f"wall time:    {elapsed:.1f}s")
print("Video saved to outputs/episode.mp4")
