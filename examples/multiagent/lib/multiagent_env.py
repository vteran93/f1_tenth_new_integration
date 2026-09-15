# Fix for gymnasium compatibility with RLlib
import math
import os
import pathlib
from collections import OrderedDict

import numpy as np
from enum import Enum
from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.reset import make_reset_fn
from f1tenth_gym.envs.track.utils import nearest_point_on_trajectory
import gymnasium as gym
import yaml
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from abc import ABC, abstractmethod


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def resolve_manifest_path(path) -> pathlib.Path:
    """Resolve a curriculum manifest path: absolute, relative to cwd, or relative to the repo root."""
    p = pathlib.Path(path)
    for cand in (p, pathlib.Path.cwd() / p, REPO_ROOT / p):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"curriculum manifest not found: {path}")


def load_curriculum_manifest(path) -> dict:
    with open(resolve_manifest_path(path)) as f:
        return yaml.safe_load(f)


class MultiAgentF110(MultiAgentEnv, ABC):
    """Multi-agent wrapper for F110Env.

    Optional ``env_config`` keys (all default to the historical behaviour when absent):

    action_repeat : int
        Number of simulator steps (``timestep`` each) per RL step. Reward is computed once
        per RL step, so progress terms naturally sum over the repeat. Default 1.
    min_speed / max_speed : float
        Clip applied to the speed command (action index 1) before it reaches the simulator.
    speed_action_range : [lo, hi]
        Bounds of the speed component of the *agent's action space* (default: the vehicle's
        v_min/v_max, i.e. [-5, 20]). With ``normalize_actions`` the policy's [-1, 1] output is
        mapped onto this range, so a range like [0.5, 8] removes the reverse/zero dead zone and
        the unreachable 8-20 m/s band that otherwise swallow ~70 % of the action range (the
        policy mean then crawls while exploration noise clipped at the caps fakes speed).
    episode_timeout : dict
        ``{"steps": N}`` or ``{"laps": L, "ref_speed": v}`` (time to drive L laps at v m/s).
        When reached the episode is truncated for all agents. Default: no time limit.
    curriculum : dict
        ``manifest`` (path to ``maps/kata_curriculum.yaml``), ``stage`` (initial 0-based
        stage), ``mix_prev`` (probability of sampling a track from an earlier stage, default
        0.25), ``use_stage_speed`` (cap speed with the stage's ``max_speed``, default true),
        ``eval_tracks`` (list of track names cycled deterministically; overrides stage
        sampling -> use it for the evaluation env). The track is (re)selected at every
        ``reset`` and the simulator map is swapped in place.
    reward_params : dict
        Free-form constants read by reward subclasses (see ``ProgressTimePenaltyEnv``).
    """

    _TRACK_CACHE_SIZE = 12

    def __init__(self, env_config=None):
        # Called when the environment is created.
        super().__init__()
        env_config = env_config or {}
        self.env = F110Env(config=env_config, render_mode=env_config.get("render_mode"))
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        self._last_positions = [(0.0, 0.0)] * self.env.num_agents
        self._crashed_agents = set()  # Track which agents have crashed

        # ---- wrapper options -------------------------------------------------------
        self.action_repeat = max(1, int(env_config.get("action_repeat", 1)))
        self._min_speed = env_config.get("min_speed")
        self._max_speed = env_config.get("max_speed")
        self._speed_action_range = env_config.get("speed_action_range")
        self._timeout_cfg = env_config.get("episode_timeout")
        self._max_episode_steps = None
        self._episode_steps = 0
        self.episode_timed_out = False

        # ---- lap bookkeeping (independent of the reward function) -------------------
        n = self.env.num_agents
        self._cum_progress = np.zeros(n)
        self._last_s_metric = np.zeros(n)

        # ---- curriculum ------------------------------------------------------------
        self.current_track = env_config.get("map")
        self.curriculum_stage = 0
        self._stage_max_speed = None
        self._curriculum = None
        self._track_cache = OrderedDict()
        cur_cfg = env_config.get("curriculum")
        if cur_cfg:
            self._init_curriculum(cur_cfg)

        # Extract single agent spaces from multi-agent F110Env
        self.action_space = self._make_single_agent_action_space()
        self.observation_space = self._make_single_agent_obs_space()

    # ------------------
    # Curriculum
    # ------------------

    def _init_curriculum(self, cfg):
        manifest = load_curriculum_manifest(cfg["manifest"])
        stages = manifest["stages"]
        self._curriculum = dict(
            stages=stages,
            mix_prev=float(cfg.get("mix_prev", 0.25)),
            use_stage_speed=bool(cfg.get("use_stage_speed", True)),
            eval_tracks=list(cfg["eval_tracks"]) if cfg.get("eval_tracks") else None,
        )
        self._eval_idx = 0
        self.set_stage(int(cfg.get("stage", 0)))

    @property
    def num_stages(self) -> int:
        return len(self._curriculum["stages"]) if self._curriculum else 1

    def set_stage(self, stage: int) -> int:
        """Select the curriculum stage used from the NEXT reset on. Returns the clamped stage."""
        if not self._curriculum:
            return 0
        stage = int(max(0, min(stage, self.num_stages - 1)))
        self.curriculum_stage = stage
        st = self._curriculum["stages"][stage]
        self._stage_max_speed = float(st["max_speed"]) if self._curriculum["use_stage_speed"] else None
        return stage

    def _sample_track(self) -> str:
        cur = self._curriculum
        if cur["eval_tracks"]:
            name = cur["eval_tracks"][self._eval_idx % len(cur["eval_tracks"])]
            self._eval_idx += 1
            return name
        stage = self.curriculum_stage
        if stage > 0 and np.random.uniform() < cur["mix_prev"]:
            stage = int(np.random.randint(0, stage))
        tracks = cur["stages"][stage]["train_tracks"]
        return tracks[int(np.random.randint(0, len(tracks)))]["name"]

    def _load_track(self, name: str) -> Track:
        if name in self._track_cache:
            self._track_cache.move_to_end(name)
            return self._track_cache[name]
        track = Track.from_track_name(name)
        self._track_cache[name] = track
        if len(self._track_cache) > self._TRACK_CACHE_SIZE:
            self._track_cache.popitem(last=False)
        return track

    def _switch_track(self, name: str):
        """Swap the simulator map in place (scan simulator, track, reset function)."""
        track = self._load_track(name)
        self.env.sim.set_map(track)
        self.env.track = track
        self.env.map = name
        self.env.reset_fn = make_reset_fn(
            **self.env.config["reset_config"], track=track, num_agents=self.env.num_agents
        )
        self.current_track = name

    @property
    def track_length(self) -> float:
        return float(self.env.track.centerline.spline.s[-1])

    @property
    def laps_completed(self) -> np.ndarray:
        """Cumulative forward progress per agent in laps (fractional, monotonic)."""
        return self._cum_progress / max(self.track_length, 1e-6)

    def _effective_max_speed(self):
        caps = [c for c in (self._max_speed, self._stage_max_speed) if c is not None]
        return min(caps) if caps else None

    def _clip_speed(self, actions: np.ndarray) -> np.ndarray:
        hi = self._effective_max_speed()
        if self._min_speed is None and hi is None:
            return actions
        actions = np.array(actions, dtype=np.float64, copy=True)
        lo = -np.inf if self._min_speed is None else self._min_speed
        actions[:, 1] = np.clip(actions[:, 1], lo, np.inf if hi is None else hi)
        return actions

    def _compute_timeout_steps(self):
        cfg = self._timeout_cfg
        if not cfg:
            return None
        if "steps" in cfg:
            return int(cfg["steps"])
        seconds = float(cfg.get("laps", 2)) * self.track_length / float(cfg.get("ref_speed", 1.5))
        return int(math.ceil(seconds / (self.env.timestep * self.action_repeat)))

    # Search window (m) around the previous arc-length when projecting the car onto the
    # centerline. A global nearest-point search flips between the two legs of a hairpin or
    # the branches of an S when the car hugs the wall, which shows up as a fake jump of
    # tens of metres in progress (and, with monotonic clipping, as farmable reward).
    ARCLENGTH_WINDOW = 6.0
    # A car cannot advance more than v_max * dt * repeat per RL step (20 m/s -> 1 m at 0.05 s).
    MAX_STEP_PROGRESS = 2.0

    def arclength(self, x: float, y: float, s_prev=None) -> float:
        """Arc-length of the centerline point nearest to (x, y), optionally restricted to a
        window around ``s_prev`` (wrap-aware)."""
        spline = self.env.track.centerline.spline
        if s_prev is None:
            return float(spline.calc_arclength_inaccurate(float(x), float(y))[0])
        n = spline.points.shape[0] - 1            # last point duplicates the first
        ds = spline.s[-1] / max(n, 1)
        half = max(2, int(self.ARCLENGTH_WINDOW / max(ds, 1e-6)))
        i0 = int(np.searchsorted(spline.s, s_prev)) % n
        lo, hi = i0 - half, i0 + half
        chunks = []
        if lo < 0:
            chunks.append(np.arange(lo + n, n))
            lo = 0
        if hi > n - 1:
            chunks.append(np.arange(0, min(hi - n + 1, n - 1) + 1))
            hi = n - 1
        chunks.append(np.arange(lo, hi + 1))
        best_s, best_d = None, np.inf
        pt = np.array([float(x), float(y)], dtype=np.float32)
        for inds in chunks:
            if len(inds) < 2:
                continue
            _, dist, t, seg = nearest_point_on_trajectory(pt, spline.points[inds, :2])
            if dist < best_d:
                k = int(inds[seg])
                best_d = dist
                best_s = float(spline.s[k] + t * (spline.s[k + 1] - spline.s[k]))
        return best_s if best_s is not None else float(spline.calc_arclength_inaccurate(float(x), float(y))[0])

    def progress_delta(self, s_new: float, s_prev: float) -> float:
        """Forward progress between two arc-lengths: wrap-aware, non-negative, bounded."""
        L = self.track_length
        d = s_new - s_prev
        if d > 0.5 * L:
            d -= L
        elif d < -0.5 * L:
            d += L
        return float(min(max(d, 0.0), self.MAX_STEP_PROGRESS))

    def _arclengths(self, use_window=False):
        return np.array([
            self.arclength(self.env.poses_x[i], self.env.poses_y[i],
                           self._last_s_metric[i] if use_window else None)
            for i in range(self.env.num_agents)
        ])

    def _update_progress_metric(self):
        s = self._arclengths(use_window=True)
        for i in range(self.env.num_agents):
            self._cum_progress[i] += self.progress_delta(s[i], self._last_s_metric[i])
        self._last_s_metric = s

    def reset(self, *, seed=None, options=None):
        # Called at the beginning of each new episode.
        if self._curriculum:
            name = self._sample_track()
            if name != self.current_track:
                self._switch_track(name)
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        self._crashed_agents = set()  # Reset crashed agents
        self._last_s = [0.0] * self.env.num_agents  # <-- Reset progress tracker
        self._episode_steps = 0
        self.episode_timed_out = False
        self._max_episode_steps = self._compute_timeout_steps()
        self._cum_progress[:] = 0.0
        self._last_s_metric = self._arclengths()

        return self._convert_obs(obs), self._convert_info(info, self.agents)

    def step(self, action_dict):
        # Called at each time step to advance the simulation.
        # Filter actions: crashed agents get zero action
        filtered_actions = []
        for i, agent in enumerate(self.agents):
            if agent in self._crashed_agents:
                # Crashed agent gets zero action (no movement)
                filtered_actions.append(np.zeros_like(self.env.action_space.low[0]))
            else:
                filtered_actions.append(action_dict.get(agent, np.zeros_like(self.env.action_space.low[0])))
        
        actions = self._clip_speed(np.asarray(filtered_actions))
        active = np.array([a not in self._crashed_agents for a in self.agents])
        for _ in range(self.action_repeat):
            obs, _, terminated, truncated, info = self.env.step(actions)
            if np.any(np.asarray(self.env.collisions, dtype=bool) & active):
                break  # a new collision: do not keep integrating the repeat
        self._episode_steps += 1
        self._update_progress_metric()
        if self._max_episode_steps is not None and self._episode_steps >= self._max_episode_steps:
            truncated = True
            self.episode_timed_out = True

        # Track newly crashed agents this step
        newly_crashed = set()
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            if self.env.collisions[i] and agent not in self._crashed_agents:
                newly_crashed.add(agent)
                self._crashed_agents.add(agent)

        # Calculate rewards and metrics
        rewards = self._get_rewards(newly_crashed)
        
        # Convert observations
        full_obs_dict = self._convert_obs(obs)
        
        # Build return dictionaries - only include active agents in obs
        obs_dict = {}
        rew_dict = {}
        terminated_dict = {}
        
        for i, agent in enumerate(self.agents):
            if agent in newly_crashed:
                # Agent just crashed this step - include final observation and reward
                obs_dict[agent] = full_obs_dict[agent]
                rew_dict[agent] = rewards[i]
                terminated_dict[agent] = True
            elif agent not in self._crashed_agents:
                # Agent is still active
                obs_dict[agent] = full_obs_dict[agent]
                rew_dict[agent] = rewards[i]
                terminated_dict[agent] = False
            # Note: Previously crashed agents are not included in any dict

        # Episode ends when ALL agents have crashed
        terminated_dict["__all__"] = len(self._crashed_agents) == len(self.agents)
        
        # Truncated dict only for active/newly crashed agents
        truncated_dict = {agent: truncated for agent in obs_dict.keys()}
        truncated_dict["__all__"] = truncated
        
        # Info dict only for active/newly crashed agents
        info_dict = self._convert_info(info, obs_dict.keys())
        
        return obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict

    def render(self):
        # Called to render the environment's current state.
        return self.env.render()

    def close(self):
        # Called to clean up resources when the environment is no longer needed.
        self.env.close()

    def _get_rewards(self, newly_crashed) -> list:
        """Iterates and computes rewards for each agent. 
        Args:
            newly_crashed (list): List of agents that crashed in this step.
        Returns:
            list: List of rewards for each agent.
        """

        # Initialize last_s tracking if not exists
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        rewards = []
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            reward = self._compute_reward(agent, newly_crashed, i)
            rewards.append(reward)

        return rewards
    
    @abstractmethod
    def _compute_reward(self, agent, newly_crashed, i) -> float:
        """Compute reward for a single agent.

        Args:
            agent (str): The ID of the agent (e.g., 'agent_0').
            newly_crashed (list): List of agents that crashed in this step.
            i (int): Index of the agent in the environment.

        Returns:
            float: The computed reward for the agent.
        """
        raise NotImplementedError("Subclasses must implement _compute_reward method.")

    # ------------------
    # Helper Methods
    # ------------------

    def _make_single_agent_action_space(self):
        # Helper to create a single-agent action space from the base env.
        """Extract single agent action space from F110Env's multi-agent action space."""
        multi_action_space = self.env.action_space
        if not isinstance(multi_action_space, gym.spaces.Box):
            raise ValueError(f"Expected Box action space, got {type(multi_action_space)}")
        single_low = np.array(multi_action_space.low[0], dtype=np.float32, copy=True)
        single_high = np.array(multi_action_space.high[0], dtype=np.float32, copy=True)
        if getattr(self, "_speed_action_range", None):
            lo, hi = self._speed_action_range
            single_low[1], single_high[1] = np.float32(lo), np.float32(hi)
        return gym.spaces.Box(low=single_low, high=single_high, shape=single_low.shape, dtype=np.float32)

    def _make_single_agent_obs_space(self):
        # Helper to create a single-agent observation space from the base env.
        """Create single agent observation space from F110Env's multi-agent space."""
        orig_spaces = self.env.observation_space.spaces
        single_spaces = {}
        for key, space in orig_spaces.items():
            if key == 'ego_idx':
                # Keep ego_idx as Discrete space (not Box)
                single_spaces[key] = space
            elif hasattr(space, 'shape') and len(space.shape) > 0 and space.shape[0] == self.env.num_agents:
                if key == 'scans':
                    # Extract single-agent scan space (create bounds directly as float32)
                    scan_shape = (space.shape[1],)
                    # Increased margin to handle sensor noise and simulation edge cases
                    low_val = np.float32(-1.0)  # Conservative margin to handle all noise outliers
                    high_val = np.float32(space.high[0].max())
                    single_spaces[key] = gym.spaces.Box(
                        low=low_val, high=high_val, shape=scan_shape, dtype=np.float32
                    )
                else:
                    # Scalar observations (create bounds directly as float32)
                    low_val = np.float32(space.low[0])
                    high_val = np.float32(space.high[0])
                    single_spaces[key] = gym.spaces.Box(
                        low=low_val, high=high_val, shape=(), dtype=np.float32
                    )
            else:
                single_spaces[key] = space
        return gym.spaces.Dict(single_spaces)

    def _convert_obs(self, obs):
        # Helper to convert the base env's observation into RLlib's per-agent format.
        """Convert multi-agent observation to per-agent format."""
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            agent_obs = {}
            for key, value in obs.items():
                if key == 'ego_idx':
                    # ego_idx is a Discrete space - pass through as-is (typically int64)
                    agent_obs[key] = value
                elif hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == self.env.num_agents:
                    # Multi-agent observation - extract for this agent and ensure correct dtype
                    agent_obs[key] = np.asarray(value[i], dtype=np.float32)
                else:
                    # Non-indexed values - ensure correct dtype
                    agent_obs[key] = np.float32(value)
            obs_dict[agent] = agent_obs
        return obs_dict

    def _convert_info(self, info, agent_keys):
        # Helper to convert the base env's info dict into RLlib's per-agent format.
        """Convert multi-agent info dict to per-agent info dicts."""
        info_dict = {}
        agent_id_to_idx = {agent_id: i for i, agent_id in enumerate(self.agents)}
        for agent in agent_keys:
            agent_idx = agent_id_to_idx[agent]
            agent_info = {}
            for k, v in info.items():
                if isinstance(v, np.ndarray) and v.shape and v.shape[0] == self.env.num_agents:
                    agent_info[k] = v[agent_idx]
                else:
                    agent_info[k] = v
            info_dict[agent] = agent_info
        return info_dict

    def _calculate_lap_progress(self):
        # Helper to calculate the normalized lap progress for each agent.
        """Calculate lap progress for each agent."""
        current_progress = []
        for i in range(self.env.num_agents):
            current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                self.env.poses_x[i], self.env.poses_y[i]
            )
            current_progress.append(current_s)
        return np.asarray([p / self.env.track.centerline.spline.s[-1] for p in current_progress])