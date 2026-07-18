#!/usr/bin/env python3
"""
Multi-Agent F1TENTH Environment Classes
Base implementation with polymorphic reward system support.
"""

import numpy as np
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from f1tenth_gym.envs import F110Env
from enum import Enum

# Fix for gymnasium compatibility with RLlib
import gymnasium.envs.registration


class VectorizeMode(Enum):
    ASYNC = "async"
    SYNC = "sync"


gymnasium.envs.registration.VectorizeMode = VectorizeMode


class MultiAgentF110(MultiAgentEnv):
    """
    Base multi-agent wrapper for F110Env.
    Implements common functionality shared between PPO and SAC implementations.
    Uses polymorphic reward system.
    """

    def __init__(self, env_config=None, reward_function=None):
        super().__init__()
        self.env = F110Env(config=env_config or {}, render_mode=env_config.get("render_mode"))
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        self._last_positions = [(0.0, 0.0)] * self.env.num_agents
        self._crashed_agents = set()

        # Polymorphic reward system
        self.reward_function = reward_function

        # Initialize tracking variables
        self._last_s = [0.0] * self.env.num_agents

    def _convert_info(self, info, agent_keys):
        """Convert multi-agent info dict to per-agent info dicts."""
        info_dict = {}
        for i, agent in enumerate(agent_keys):
            agent_info = {}
            for k, v in info.items():
                if isinstance(v, np.ndarray) and v.shape and v.shape[0] == self.env.num_agents:
                    agent_info[k] = v[i]
                else:
                    agent_info[k] = v
            info_dict[agent] = agent_info
        return info_dict

    def reset(self, *, seed=None, options=None):
        """Reset environment and initialize tracking variables."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        self._crashed_agents = set()
        self._last_s = [0.0] * self.env.num_agents

        return self._convert_obs(obs), self._convert_info(info, self.agents)

    def step(self, action_dict):
        """Step environment with action filtering for crashed agents."""
        # Filter actions: crashed agents get zero action
        filtered_actions = []
        for i, agent in enumerate(self.agents):
            if agent in self._crashed_agents:
                filtered_actions.append(np.zeros_like(self.env.action_space.low[0]))
            else:
                filtered_actions.append(action_dict.get(agent, np.zeros_like(self.env.action_space.low[0])))

        actions = np.array(filtered_actions)
        obs, _, terminated, truncated, info = self.env.step(actions)

        # Track newly crashed agents this step
        newly_crashed = set()
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            if self.env.collisions[i] and agent not in self._crashed_agents:
                newly_crashed.add(agent)
                self._crashed_agents.add(agent)

        # Calculate rewards using polymorphic reward system
        if self.reward_function:
            rewards = self.reward_function._get_rewards(self, newly_crashed)
        else:
            rewards = self._get_default_rewards(newly_crashed)

        # Convert observations
        full_obs_dict = self._convert_obs(obs)

        # Build return dictionaries - only include active agents in obs
        obs_dict = {}
        rew_dict = {}
        terminated_dict = {}

        for i, agent in enumerate(self.agents):
            if agent not in self._crashed_agents:
                obs_dict[agent] = full_obs_dict[agent]
                rew_dict[agent] = rewards[i]
                terminated_dict[agent] = False
            elif agent in newly_crashed:
                obs_dict[agent] = full_obs_dict[agent]
                rew_dict[agent] = rewards[i]
                terminated_dict[agent] = True

        # Episode ends when ALL agents have crashed
        terminated_dict["__all__"] = len(self._crashed_agents) == len(self.agents)

        # Truncated dict only for active/newly crashed agents
        truncated_dict = {agent: truncated for agent in obs_dict.keys()}
        truncated_dict["__all__"] = truncated

        # Info dict only for active/newly crashed agents
        info_dict = self._convert_info(info, obs_dict.keys())

        return obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict

    def _get_default_rewards(self, newly_crashed):
        """Default reward function if no polymorphic reward is provided."""
        rewards = []
        for i in range(self.env.num_agents):
            agent = self.agents[i]

            if agent in self._crashed_agents and agent not in newly_crashed:
                reward = 0.0
            else:
                # Simple progress-based reward
                current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                    self.env.poses_x[i], self.env.poses_y[i]
                )
                prog = current_s - self._last_s[i]

                if prog > 0.9 * self.env.track.centerline.spline.s[-1]:
                    prog = (self.env.track.centerline.spline.s[-1] - self._last_s[i]) + current_s

                reward = prog
                if agent in newly_crashed:
                    reward -= 1.0

                self._last_s[i] = current_s

            rewards.append(reward)

        return rewards

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # Abstract methods to be implemented by subclasses
    def _convert_obs(self, obs):
        """Convert observations. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _convert_obs")

    def _make_single_agent_obs_space(self):
        """Create observation space. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _make_single_agent_obs_space")

    def _make_single_agent_action_space(self):
        """Create action space. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _make_single_agent_action_space")


class MultiAgentF110PPO(MultiAgentF110):
    """
    PPO-specific implementation of MultiAgentF110.
    Features normalized observations and specific action/observation spaces.
    """

    def __init__(self, env_config=None, reward_function=None):
        super().__init__(env_config, reward_function)

        # PPO-specific setup: Dynamic Normalization
        self.vehicle_params = self.env.params
        self.max_scan_range = np.max(self.env.observation_space.spaces['scans'].high)
        self.map_x_min = np.min(self.env.track.centerline.xs)
        self.map_x_max = np.max(self.env.track.centerline.xs)
        self.map_y_min = np.min(self.env.track.centerline.ys)
        self.map_y_max = np.max(self.env.track.centerline.ys)

        # Extract single agent spaces from multi-agent F110Env
        self.action_space = self._make_single_agent_action_space()
        self.observation_space = self._make_single_agent_obs_space()

    def _make_single_agent_obs_space(self):
        """Create a single agent's observation space with normalized bounds."""
        return gym.spaces.Dict({
            "ego_idx": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "scans": gym.spaces.Box(low=0.0, high=1.0, shape=(self.env.num_beams,), dtype=np.float32),
            "poses_x": gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            "poses_y": gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            "poses_theta": gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            "linear_vels_x": gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            "linear_vels_y": gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            "ang_vels_z": gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32),
            "collisions": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })

    def _convert_obs(self, obs):
        """Convert multi-agent observation to per-agent format and apply normalization."""
        obs_dict = {}
        v_min, v_max = self.vehicle_params['v_min'], self.vehicle_params['v_max']
        sv_max = self.vehicle_params['sv_max']

        for i, agent in enumerate(self.agents):
            agent_obs = {}

            agent_obs["ego_idx"] = float(i) / (self.env.num_agents - 1) if self.env.num_agents > 1 else 0.0
            agent_obs["scans"] = np.clip(obs["scans"][i] / self.max_scan_range, 0.0, 1.0)
            agent_obs["poses_x"] = 2 * (obs["poses_x"][i] - self.map_x_min) / (self.map_x_max - self.map_x_min) - 1
            agent_obs["poses_y"] = 2 * (obs["poses_y"][i] - self.map_y_min) / (self.map_y_max - self.map_y_min) - 1
            agent_obs["poses_theta"] = obs["poses_theta"][i] / np.pi
            agent_obs["linear_vels_x"] = 2 * (obs["linear_vels_x"][i] - v_min) / (v_max - v_min) - 1
            agent_obs["linear_vels_y"] = 2 * (obs["linear_vels_y"][i] - v_min) / (v_max - v_min) - 1
            agent_obs["ang_vels_z"] = np.clip(obs["ang_vels_z"][i] / sv_max, -1.0, 1.0)
            agent_obs["collisions"] = obs["collisions"][i].astype(float)

            for key, value in agent_obs.items():
                agent_obs[key] = np.array(value, dtype=np.float32)
                if key in ["scans", "ego_idx", "collisions"]:
                    agent_obs[key] = np.clip(agent_obs[key], 0.0, 1.0)
                else:
                    agent_obs[key] = np.clip(agent_obs[key], -1.0, 1.0)

            obs_dict[agent] = agent_obs
        return obs_dict

    def _make_single_agent_action_space(self):
        """Extract single agent action space from F110Env's multi-agent action space."""
        multi_action_space = self.env.action_space

        if not isinstance(multi_action_space, gym.spaces.Box):
            raise ValueError(f"Expected Box action space, got {type(multi_action_space)}")

        single_low = multi_action_space.low[0]
        single_high = multi_action_space.high[0]

        return gym.spaces.Box(
            low=single_low,
            high=single_high,
            shape=single_low.shape,
            dtype=np.float32
        )


class MultiAgentF110SAC(MultiAgentF110):
    """
    SAC-specific implementation of MultiAgentF110.
    Features different observation space handling and action space structure.
    """

    def __init__(self, env_config=None, reward_function=None):
        super().__init__(env_config, reward_function)

        # SAC-specific setup: Fixed action spaces
        self.action_space = {agent: gym.spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 10.0]), dtype=np.float32)
                             for agent in self.agents}
        self.observation_space = self._make_single_agent_obs_space()

    def _make_single_agent_obs_space(self):
        """Create single agent observation space from F110Env's multi-agent space."""
        orig_spaces = self.env.observation_space.spaces
        single_spaces = {}

        for key, space in orig_spaces.items():
            if key == 'ego_idx':
                single_spaces[key] = gym.spaces.Box(low=0, high=space.n-1, shape=(), dtype=np.int32)
            elif hasattr(space, 'shape') and len(space.shape) > 0 and space.shape[0] == self.env.num_agents:
                if key == 'scans':
                    single_spaces[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(),
                                                        shape=(space.shape[1],), dtype=space.dtype)
                else:
                    single_spaces[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(),
                                                        shape=(), dtype=space.dtype)
            else:
                single_spaces[key] = space
        return {agent: gym.spaces.Dict(single_spaces) for agent in self.agents}

    def _convert_obs(self, obs):
        """Convert multi-agent observation to per-agent format."""
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            agent_obs = {}
            for key, value in obs.items():
                if key == 'ego_idx':
                    agent_obs[key] = np.array(value, dtype=np.int32)
                elif hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == self.env.num_agents:
                    original_space = self.env.observation_space.spaces[key]
                    agent_obs[key] = np.clip(
                        value[i].astype(original_space.dtype),
                        original_space.low.min(),
                        original_space.high.max()
                    )
                else:
                    agent_obs[key] = np.array(value, dtype=value.dtype if hasattr(value, 'dtype') else np.float32)

            obs_dict[agent] = agent_obs
        return obs_dict

    def _convert_info(self, info, agent_keys):
        """Convert info dict for SAC (simpler version)."""
        return {agent: info for agent in agent_keys}

    def _make_single_agent_action_space(self):
        """SAC uses predefined action space structure."""
        return gym.spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 10.0]), dtype=np.float32)
