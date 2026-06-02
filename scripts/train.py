#!/usr/bin/env python3
"""
Train a PPO agent on CommuteEnv and compare it against the naive baseline.

Usage:
    .venv/bin/python3 scripts/train.py

Outputs:
    outputs/ppo_commute.zip   — saved SB3 model (reload with PPO.load())
    outputs/vecnorm.pkl       — observation/reward normalisation stats
"""
import os
import time
import warnings
import numpy as np

# Suppress the "no path" warnings that spam output during training.
warnings.filterwarnings("ignore", category=UserWarning, message=".*no path.*")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from transport_brain.network import load_network
from transport_brain.env import CommuteEnv


# ── Config ────────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 2_000_000  # ~8300 episodes; adjust up for better convergence
N_ENVS = 4                  # parallel envs (DummyVecEnv, no multiprocessing)
MODEL_PATH = "outputs/ppo_commute.zip"
VECNORM_PATH = "outputs/vecnorm.pkl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env(seed=0):
    net, _node_ids, node_xy, attractors = load_network("data/cph.graphml")
    def _init():
        env = CommuteEnv(
            net=net,
            node_xy=node_xy,
            attractors=attractors,
            n_trips=4000,
            n_steps=240,
            max_release=20,
            seed=seed,
        )
        return env
    return _init


def _episode_delay_veh_hours(env, n_steps):
    """Compute delay for a completed episode, excluding never-departed vehicles."""
    from transport_brain.dynamic_sim import DT
    sim = env._sim
    started = sim.started
    # Cap started-but-not-arrived vehicles (arrival_step is still 0 otherwise).
    started_not_arrived = started & ~sim.arrived
    if started_not_arrived.any():
        sim.arrival_step[started_not_arrived] = n_steps
    if not started.any():
        return 0.0
    tt = float(np.sum((sim.arrival_step[started] - sim.departure_step[started]) * DT))
    delay_s = tt - float(sim.free_flow_time_s[started].sum())
    return delay_s / 3600


def evaluate(model, vec_env, n_episodes=3):
    """Run n_episodes on a fresh single env with the trained policy."""
    net, _node_ids, node_xy, attractors = load_network("data/cph.graphml")
    delays = []
    for ep in range(n_episodes):
        env = CommuteEnv(
            net=net, node_xy=node_xy, attractors=attractors,
            n_trips=4000, n_steps=240, max_release=20, seed=100 + ep,
        )
        obs, _ = env.reset()
        # Normalise observation using the trained VecNormalize stats.
        obs_n = vec_env.normalize_obs(np.array([obs]))[0]
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs_n[np.newaxis], deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action[0]))
            obs_n = vec_env.normalize_obs(np.array([obs]))[0]
        delays.append(_episode_delay_veh_hours(env, env.n_steps))
    return float(np.mean(delays))


def naive_baseline(n_episodes=3):
    """Run the constant release-10 policy; return mean delay in veh-hours."""
    net, _node_ids, node_xy, attractors = load_network("data/cph.graphml")
    delays = []
    for ep in range(n_episodes):
        env = CommuteEnv(
            net=net, node_xy=node_xy, attractors=attractors,
            n_trips=4000, n_steps=240, max_release=20, seed=ep,
        )
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, _ = env.step(10)
        delays.append(_episode_delay_veh_hours(env, env.n_steps))
    return float(np.mean(delays))


class ProgressCallback(BaseCallback):
    """Print episode reward and delay every N steps."""

    def __init__(self, print_every=10_000):
        super().__init__()
        self.print_every = print_every
        self._next_print = print_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_print:
            # Mean episode reward from SB3's rolling buffer.
            if len(self.model.ep_info_buffer) > 0:
                mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                print(
                    f"  step {self.num_timesteps:>7,} | "
                    f"mean_ep_reward {mean_r:+.4f} | "
                    f"mean_ep_len {mean_len:.0f}"
                )
            self._next_print += self.print_every
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("Loading Copenhagen network…")
    # Measure naive baseline first so we have a target.
    print("Computing naive baseline (3 episodes)…")
    baseline_delay = naive_baseline(n_episodes=3)
    print(f"  Naive policy (release 10/step): {baseline_delay:.1f} veh-hours delay\n")

    print(f"Building {N_ENVS} parallel envs…")
    vec_env = DummyVecEnv([make_env(seed=i) for i in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps…")
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
    )

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=ProgressCallback(print_every=10_000),
        progress_bar=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTraining done in {elapsed:.0f}s.")

    # Save model + normalisation stats.
    model.save(MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Evaluate trained agent.
    print("\nEvaluating trained agent (3 episodes)…")
    trained_delay = evaluate(model, vec_env, n_episodes=3)
    print(f"  Naive policy:   {baseline_delay:.1f} veh-hours")
    print(f"  Trained agent:  {trained_delay:.1f} veh-hours")
    if trained_delay < baseline_delay:
        pct = (baseline_delay - trained_delay) / baseline_delay * 100
        print(f"  Improvement:    {pct:.1f}%")
    else:
        print("  Agent did not improve over naive baseline yet — train longer.")
