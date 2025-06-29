# Fix for gymnasium compatibility with RLlib
import numpy as np
from enum import Enum
from f1tenth_gym.envs import F110Env
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from abc import ABC, abstractmethod

# Fix for gymnasium compatibility with RLlib
import gymnasium.envs.registration

class VectorizeMode(Enum):
    ASYNC = "async"
    SYNC = "sync"

gymnasium.envs.registration.VectorizeMode = VectorizeMode


class MultiAgentF110(MultiAgentEnv, ABC):
    """Multi-agent wrapper for F110Env."""

    def __init__(self, env_config=None):
        # Called when the environment is created.
        super().__init__()
        self.env = F110Env(config=env_config or {}, render_mode=env_config.get("render_mode"))
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        self._last_positions = [(0.0, 0.0)] * self.env.num_agents
        self._crashed_agents = set()  # Track which agents have crashed
        
        # Extract single agent spaces from multi-agent F110Env
        self.action_space = self._make_single_agent_action_space()
        self.observation_space = self._make_single_agent_obs_space()

    def reset(self, *, seed=None, options=None):
        # Called at the beginning of each new episode.
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        self._crashed_agents = set()  # Reset crashed agents
        self._last_s = [0.0] * self.env.num_agents  # <-- Reset progress tracker

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
        
        actions = np.array(filtered_actions)
        obs, _, terminated, truncated, info = self.env.step(actions)

        # Track newly crashed agents this step
        newly_crashed = set()
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            if self.env.collisions[i] and agent not in self._crashed_agents:
                newly_crashed.add(agent)
                self._crashed_agents.add(agent)

        
        # Calculate rewards and metrics
        rewards = self._get_rewards(newly_crashed)
        lap_progress = self._calculate_lap_progress()
        info["lap_progress"] = lap_progress

        # Convert observations
        full_obs_dict = self._convert_obs(obs)
        
        # Build return dictionaries - only include active agents in obs
        obs_dict = {}
        rew_dict = {}
        terminated_dict = {}
        
        for i, agent in enumerate(self.agents):
            if agent not in self._crashed_agents:
                # Agent is still active
                obs_dict[agent] = full_obs_dict[agent]
                rew_dict[agent] = rewards[i]
                terminated_dict[agent] = False
            elif agent in newly_crashed:
                # Agent just crashed this step - include final observation and reward
                obs_dict[agent] = full_obs_dict[agent]
                rew_dict[agent] = rewards[i]
                terminated_dict[agent] = True
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

    @abstractmethod
    def _get_rewards(self, newly_crashed) -> list:
        # Abstract method to be implemented by subclasses for reward calculation.
        ''' Needs to be implemented by inheriting this class.
         Returns a list of rewards for each agent based on the current state of the environment.'''
        pass

    # ------------------
    # Helper Methods
    # ------------------

    def _make_single_agent_action_space(self):
        # Helper to create a single-agent action space from the base env.
        """Extract single agent action space from F110Env's multi-agent action space."""
        multi_action_space = self.env.action_space
        if not isinstance(multi_action_space, gym.spaces.Box):
            raise ValueError(f"Expected Box action space, got {type(multi_action_space)}")
        single_low = multi_action_space.low[0]
        single_high = multi_action_space.high[0]
        return gym.spaces.Box(low=single_low, high=single_high, shape=single_low.shape, dtype=np.float32)

    def _make_single_agent_obs_space(self):
        # Helper to create a single-agent observation space from the base env.
        """Create single agent observation space from F110Env's multi-agent space."""
        orig_spaces = self.env.observation_space.spaces
        single_spaces = {}
        for key, space in orig_spaces.items():
            if key == 'ego_idx':
                single_spaces[key] = gym.spaces.Box(low=0, high=space.n-1, shape=(), dtype=np.int32)
            elif hasattr(space, 'shape') and len(space.shape) > 0 and space.shape[0] == self.env.num_agents:
                if key == 'scans':
                    single_spaces[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(), shape=(space.shape[1],), dtype=space.dtype)
                else:
                    single_spaces[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(), shape=(), dtype=space.dtype)
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
                    agent_obs[key] = np.array(value, dtype=np.int32)
                elif hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == self.env.num_agents:
                    original_space = self.env.observation_space.spaces[key]
                    agent_obs[key] = np.clip(value[i].astype(original_space.dtype), original_space.low.min(), original_space.high.max())
                else:
                    agent_obs[key] = np.array(value, dtype=value.dtype if hasattr(value, 'dtype') else np.float32)
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
        return np.array([p / self.env.track.centerline.spline.s[-1] for p in current_progress])