#!/usr/bin/env python
"""Deterministic per-track evaluation of a kata-curriculum checkpoint.

Loads ONLY the policy (no Ray cluster, no env runners), builds the evaluation env of the
experiment for one track at a time and runs N episodes with explore=False. Reports, per
track and agent-episode: laps completed (monotonic progress), crash, time-out, mean speed,
and lap time of the first full lap when there is one.

Usage (from examples/multiagent):
  ../../venv/bin/python kata_eval_tracks.py <checkpoint_dir> [--episodes 3] [--tracks a,b,...]
        [--config configs/experiments_kata.yaml --experiment kata_PPO_shared_ProgressTimePenalty]
        [--csv out.csv] [--max-speed 8]
"""
import argparse
import csv
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from lib.utils import load_config, get_reward_class, find_experiment  # noqa: E402


def flattener(policy):
    """Return a function mapping a raw (dict) observation to the policy's input."""
    from ray.rllib.models.preprocessors import get_preprocessor
    space = getattr(policy.observation_space, "original_space", None) or policy.observation_space
    prep = get_preprocessor(space)(space)
    return lambda obs: prep.transform(obs)


def actor(policy):
    """Deterministic action in ENV units. Policy.compute_single_action returns the policy's
    normalized action in Ray 2.46 (unsquashing lives in the RolloutWorker), so undo it here."""
    from ray.rllib.utils.spaces.space_utils import unsquash_action, clip_action
    normalize = bool(policy.config.get("normalize_actions", False))
    clip = bool(policy.config.get("clip_actions", False))

    def act(flat_obs):
        a, _, _ = policy.compute_single_action(flat_obs, explore=False)
        if normalize:
            a = unsquash_action(a, policy.action_space_struct)
        elif clip:
            a = clip_action(a, policy.action_space_struct)
        return a
    return act


def run_track(env, policy, prep, track, episodes, act=None):
    act = act or actor(policy)
    env._curriculum["eval_tracks"] = [track]
    env._eval_idx = 0
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset()
        assert env.current_track == track, env.current_track
        done, steps, speeds = False, 0, {a: [] for a in env.agents}
        first_lap_step = {a: None for a in env.agents}
        while not done:
            actions = {}
            for a, o in obs.items():
                actions[a] = act(prep(o))
            obs, rew, term, trunc, _ = env.step(actions)
            steps += 1
            for i, a in enumerate(env.agents):
                if a not in env._crashed_agents:
                    speeds[a].append(float(env.env.sim.agents[i].state[3]))
                if first_lap_step[a] is None and env.laps_completed[i] >= 1.0:
                    first_lap_step[a] = steps
            done = term["__all__"] or trunc["__all__"]
        dt = env.env.timestep * env.action_repeat
        for i, a in enumerate(env.agents):
            rows.append(dict(
                track=track, episode=ep, agent=a,
                laps=round(float(env.laps_completed[i]), 3),
                crashed=int(a in env._crashed_agents),
                timed_out=int(env.episode_timed_out),
                seconds=round(steps * dt, 1),
                mean_speed=round(float(np.mean(speeds[a])) if speeds[a] else 0.0, 2),
                first_lap_s=round(first_lap_step[a] * dt, 1) if first_lap_step[a] else None,
                track_len=round(env.track_length, 1),
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--config", default="configs/experiments_kata.yaml")
    ap.add_argument("--experiment", default="kata_PPO_shared_ProgressTimePenalty")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--tracks", default=None, help="comma list; default = experiment's eval_tracks")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--max-speed", type=float, default=None, help="cap applied in the env (default: none, like eval)")
    args = ap.parse_args()

    from ray.rllib.policy.policy import Policy
    from run import register_nan_protected_action_dist  # policies trained with nan_protection need it
    register_nan_protected_action_dist()
    import os
    pol_dir = os.path.abspath(args.checkpoint.rstrip("/")) + "/policies/shared_policy"
    policy = Policy.from_checkpoint(pol_dir)
    prep = flattener(policy)

    cfg = load_config(args.config)
    exp = find_experiment(cfg["experiments"], args.experiment)
    env_cfg = dict(exp["evaluation"]["env"])
    if args.max_speed is not None:
        env_cfg["max_speed"] = args.max_speed
    tracks = args.tracks.split(",") if args.tracks else list(env_cfg["curriculum"]["eval_tracks"])
    env_cfg["curriculum"] = dict(env_cfg["curriculum"], eval_tracks=tracks[:1])
    env = get_reward_class({"training": exp["training"]})(env_config=env_cfg)

    all_rows = []
    print(f"{'track':34s} {'L(m)':>5} {'laps':>5} {'success':>7} {'crash':>5} {'tout':>4} {'speed':>5} {'lap1(s)':>7}")
    for t in tracks:
        rows = run_track(env, policy, prep, t, args.episodes)
        all_rows += rows
        laps = np.array([r["laps"] for r in rows])
        lap1 = [r["first_lap_s"] for r in rows if r["first_lap_s"]]
        print(f"{t:34s} {rows[0]['track_len']:5.0f} {laps.mean():5.2f} {np.mean(laps >= 1):7.2f} "
              f"{np.mean([r['crashed'] for r in rows]):5.2f} {np.mean([r['timed_out'] for r in rows]):4.2f} "
              f"{np.mean([r['mean_speed'] for r in rows]):5.2f} {np.mean(lap1) if lap1 else float('nan'):7.1f}")
        sys.stdout.flush()
    laps = np.array([r["laps"] for r in all_rows])
    print(f"\nOVERALL: success={np.mean(laps >= 1):.2f} laps={laps.mean():.2f} "
          f"crash={np.mean([r['crashed'] for r in all_rows]):.2f} n={len(all_rows)} agent-episodes")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print("csv:", args.csv)


if __name__ == "__main__":
    main()
