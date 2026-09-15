#!/usr/bin/env python
"""Render MP4 videos of a checkpoint's deterministic policy on given kata tracks.

Headless (SDL dummy driver). One video per track; frames are captured once per RL step
(0.05 s with action_repeat 5), so the video plays in real time at 20 fps. A HUD shows the
track, elapsed time, per-agent speed and laps.

Usage (from examples/multiagent):
  ../../venv/bin/python kata_render_video.py <checkpoint_dir> --tracks kata_01_taikyoku_shodan_e1,... \
        [--seconds 30] [--outdir eval_videos/kata_20260912] [--experiment kata_PPO_shared_ProgressTimePenalty]
"""
import argparse
import os
import pathlib
import warnings

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
warnings.filterwarnings("ignore")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lib.utils import load_config, get_reward_class, find_experiment  # noqa: E402
from kata_eval_tracks import flattener, actor  # noqa: E402


def hud(frame, lines):
    frame = np.ascontiguousarray(frame)
    y = 48  # below the renderer's own header line
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return frame


def render_track(policy, prep, act, exp, track, seconds, out_path, label):
    env_cfg = dict(exp["evaluation"]["env"])
    env_cfg["map"] = track                        # initial map == track: no hot swap, renderer stays valid
    env_cfg["curriculum"] = dict(env_cfg["curriculum"], eval_tracks=[track])
    env_cfg["render_mode"] = "rgb_array"
    env = get_reward_class({"training": exp["training"]})(env_config=env_cfg)
    dt = env.env.timestep * env.action_repeat
    max_steps = int(seconds / dt)
    obs, _ = env.reset()
    frames, done, step = [], False, 0
    while not done and step < max_steps:
        actions = {a: act(prep(o)) for a, o in obs.items()}
        obs, _, term, trunc, _ = env.step(actions)
        step += 1
        frame = env.render()
        if frame is not None:
            speeds = [float(env.env.sim.agents[i].state[3]) for i in range(env.env.num_agents)]
            laps = env.laps_completed
            lines = [f"{label}  {track}", f"t={step * dt:5.1f}s  L={env.track_length:.0f} m"]
            for i in range(env.env.num_agents):
                st = "CRASH" if env.agents[i] in env._crashed_agents else f"{speeds[i]:4.1f} m/s"
                lines.append(f"agent_{i}: {st}  laps={laps[i]:.2f}")
            frames.append(hud(frame, lines))
        done = term["__all__"] or trunc["__all__"]
    env.close()
    if not frames:
        print("no frames for", track)
        return None
    fps = int(round(1.0 / dt))
    import imageio.v2 as imageio
    h, w = frames[0].shape[:2]
    frames = [f[: h - h % 2, : w - w % 2] for f in frames]   # libx264 needs even dimensions
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264", quality=7, macro_block_size=None)
    print(f"{track}: {len(frames)} frames ({len(frames) / fps:.1f} s) laps={np.round(env.laps_completed, 2)} "
          f"crashed={sorted(env._crashed_agents)} -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--outdir", default="eval_videos/kata")
    ap.add_argument("--config", default="configs/experiments_kata.yaml")
    ap.add_argument("--experiment", default="kata_PPO_shared_ProgressTimePenalty")
    ap.add_argument("--label", default="PPO final")
    args = ap.parse_args()

    from ray.rllib.policy.policy import Policy
    from run import register_nan_protected_action_dist
    register_nan_protected_action_dist()
    policy = Policy.from_checkpoint(os.path.abspath(args.checkpoint.rstrip("/")) + "/policies/shared_policy")
    prep, act = flattener(policy), actor(policy)
    cfg = load_config(args.config)
    exp = find_experiment(cfg["experiments"], args.experiment)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for t in args.tracks.split(","):
        render_track(policy, prep, act, exp, t, args.seconds, outdir / f"{t}.mp4", args.label)


if __name__ == "__main__":
    main()
