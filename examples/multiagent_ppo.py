#!/usr/bin/env python3
"""
Simplified F1TENTH Multi-Agent Training/Evaluation Script
Supports both training and evaluation with minimal configuration.
"""

import numpy as np
import ray
from ray.tune.logger import UnifiedLogger
import gymnasium as gym
import tempfile
from datetime import datetime
import os
import time
import argparse
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from f1tenth_gym.envs import F110Env
from enum import Enum
import torch

# Fix for gymnasium compatibility with RLlib
import gymnasium.envs.registration


class VectorizeMode(Enum):
    ASYNC = "async"
    SYNC = "sync"


gymnasium.envs.registration.VectorizeMode = VectorizeMode


class MultiAgentF110(MultiAgentEnv):
    """Simplified multi-agent wrapper for F110Env."""

    def __init__(self, env_config=None):
        super().__init__()
        self.env = F110Env(config=env_config or {}, render_mode=env_config.get("render_mode"))
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        self._last_positions = [(0.0, 0.0)] * self.env.num_agents
        self._crashed_agents = set()

        # --- Dynamic Normalization Setup ---
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
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        self._crashed_agents = set()  # Reset crashed agents
        self._last_s = [0.0] * self.env.num_agents  # <-- Reset progress tracker

        return self._convert_obs(obs), self._convert_info(info, self.agents)

    def step(self, action_dict):
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

        # Calculate rewards
        rewards = self._get_rewards(newly_crashed)

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

    def _get_rewards(self, newly_crashed):
        """Calculate individual rewards for each agent based on F110Env reward function."""

        # Initialize last_s tracking if not exists (track progress for each agent)
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        rewards = []
        for i in range(self.env.num_agents):
            agent = self.agents[i]

            if agent in self._crashed_agents and agent not in newly_crashed:
                # Agent was already crashed - no reward calculation needed
                reward = 0.0
            else:
                # Calculate track progress using centerline spline (from F110Env)
                current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                    self.env.poses_x[i], self.env.poses_y[i]
                )

                # Calculate progress since last step
                prog = current_s - self._last_s[i]

                # Handle lap completion (when current_s wraps around to beginning)
                if prog > 0.9 * self.env.track.centerline.spline.s[-1]:
                    prog = (self.env.track.centerline.spline.s[-1] - self._last_s[i]) + current_s

                # Start with progress reward (main component from F110Env)
                reward = prog

                # Apply collision penalty (from F110Env)
                if agent in newly_crashed:  # Only penalize when agent crashes this step
                    reward -= 1.0

                # Update last track position for this agent
                self._last_s[i] = current_s

            rewards.append(reward)

        return rewards

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def _make_single_agent_action_space(self):
        """Extract single agent action space from F110Env's multi-agent action space."""
        # F110Env action space is (num_agents, action_dim)
        # We extract the first agent's action space bounds
        multi_action_space = self.env.action_space

        # Ensure it's a Box space as expected
        if not isinstance(multi_action_space, gym.spaces.Box):
            raise ValueError(f"Expected Box action space, got {type(multi_action_space)}")

        # Extract single agent bounds from multi-agent space
        single_low = multi_action_space.low[0]  # First agent's lower bounds
        single_high = multi_action_space.high[0]  # First agent's upper bounds

        return gym.spaces.Box(
            low=single_low,
            high=single_high,
            shape=single_low.shape,
            dtype=np.float32
        )


def get_env_config(render_mode=None):
    """Get environment configuration."""
    return {
        "map": "Spielberg",
        "num_agents": 2,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "original"},
        "reset_config": {"type": "cl_grid_static"},
        "render_mode": render_mode,
    }


def setup_policies_and_config():
    """Setup policies and PPO configuration. Returns (policies, config)."""
    register_env("f1tenth_multi", lambda config: MultiAgentF110(config))

    # Create temporary environment for policy setup
    temp_env = MultiAgentF110(get_env_config())
    policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
                for agent in temp_env.agents}
    temp_env.close()

    # Configure PPO
    config = (PPOConfig()
              .environment("f1tenth_multi", env_config=get_env_config())
              .framework("torch")
              .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
              .env_runners(num_env_runners=0)
              .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
              .training(
                  train_batch_size=4000,
                  # 🏗️ CONFIGURACIÓN DE RED NEURONAL PARA PPO
                  model={
                      "fcnet_hiddens": [256, 256],        # Capas ocultas [neurona_capa1, neurona_capa2]
                      "fcnet_activation": "relu",         # Función de activación
                      "vf_share_layers": True,           # Compartir capas entre value function y policy
                  }
    )
    )
    return policies, config


def custom_log_creator(custom_path: str, prefix: str):
    """
    Returns a function __logger_creator(config) that will
    write TensorBoard event files to `custom_path`.
    """
    def logger_creator(config):
        os.makedirs(custom_path, exist_ok=True)
        # Create a unique subfolder per run
        run_dir = tempfile.mkdtemp(
            prefix=f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_",
            dir=custom_path
        )
        return UnifiedLogger(config, run_dir, loggers=None)
    return logger_creator


def setup_ray_and_algo(config):
    """Initialize Ray and build algorithm—with TensorBoard logging."""
    ray.init(ignore_reinit_error=True)
    # Create a per-run log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(f"runs/multiagent_ppo_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Build with a custom logger_creator to dump TB event files there
    return config.build(
        logger_creator=custom_log_creator(log_dir, "rllib_run")
    )


def setup_training():
    """Setup and run training."""
    print("Starting training...")

    # Setup
    timestamp = str(int(time.time()))
    model_dir = f"models/multiagent_ppo_run_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # Setup policies and config
    policies, config = setup_policies_and_config()
    algo = setup_ray_and_algo(config)

    # Training loop
    TOTAL_TIMESTEPS = 500_000
    while True:
        result = algo.train()
        timesteps_total = result['timesteps_total']
        print(f"Timesteps: {timesteps_total}")
        algo.save(model_dir)

        if timesteps_total >= TOTAL_TIMESTEPS:
            break

    final_checkpoint = algo.save(model_dir)
    print(f"Training completed. Model saved to {final_checkpoint}")
    algo.stop()


def setup_evaluation():
    """Setup and run evaluation."""
    print("Starting evaluation...")

    # Find latest model
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found. Train a model first.")
        return

    run_dirs = [d for d in os.listdir(models_dir) if d.startswith("multiagent_ppo_run_")]
    if not run_dirs:
        print("No trained models found. Train a model first.")
        return

    latest_model = max(run_dirs, key=lambda x: int(x.split("_")[-1]))
    model_path = os.path.abspath(os.path.join(models_dir, latest_model))
    print(f"Using model: {latest_model}")

    # Setup algorithm
    policies, config = setup_policies_and_config()
    algo = setup_ray_and_algo(config)
    algo.restore(model_path)

    # Run evaluation
    eval_env = MultiAgentF110(get_env_config(render_mode="human"))

    for episode in range(3):
        obs_dict, _ = eval_env.reset(seed=episode)
        done = False
        step_count = 0

        while not done and step_count < 100000:
            action_dict = {agent_id: algo.compute_single_action(obs, policy_id=agent_id, explore=False)
                           for agent_id, obs in obs_dict.items()}

            obs_dict, rew_dict, terminated_dict, truncated_dict, _ = eval_env.step(action_dict)
            eval_env.render()
            done = terminated_dict["__all__"] or truncated_dict["__all__"]
            step_count += 1

        print(f"Episode {episode + 1} completed in {step_count} steps")

    eval_env.close()
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1TENTH Multi-Agent RL Training/Evaluation")
    parser.add_argument("--train", action="store_true", help="Run training mode instead of evaluation")
    args = parser.parse_args()

    if args.train:
        setup_training()
    else:
        setup_evaluation()

    ray.shutdown()
