#!/usr/bin/env python3
"""
Simplified F1TENTH Multi-Agent SAC Training/Evaluation Script
Supports both training and evaluation with minimal configuration using SAC algorithm.
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
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from f1tenth_gym.envs import F110Env
from enum import Enum
import torch.nn as nn

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

        # Define single agent spaces - simplify observation space for RLlib compatibility
        single_action_space = gym.spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 10.0]), dtype=np.float32)
        single_obs_space = self._make_simplified_obs_space()

        # Create multi-agent spaces as dictionaries
        self.action_space = {agent: single_action_space for agent in self.agents}
        self.observation_space = {agent: single_obs_space for agent in self.agents}

    def _make_simplified_obs_space(self):
        """Create a simplified observation space that RLlib can handle easily."""
        # Use only the scan data as it's the most important for racing
        return gym.spaces.Box(
            low=0.0,
            high=30.5,
            shape=(36,),  # Just the LIDAR scans
            dtype=np.float32
        )

    def _convert_obs(self, obs):
        """Convert multi-agent observation to simplified per-agent format."""
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            # Use only the LIDAR scans for simplicity
            scans = obs['scans'][i] if 'scans' in obs else np.zeros(36, dtype=np.float32)
            obs_dict[agent] = scans.astype(np.float32)
        return obs_dict

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        return self._convert_obs(obs), {agent: info for agent in self.agents}

    def step(self, action_dict):
        actions = np.array([action_dict[agent] for agent in self.agents])
        obs, _, terminated, truncated, info = self.env.step(actions)

        # Calculate rewards
        rewards = []
        for i in range(self.env.num_agents):
            pos = (self.env.poses_x[i], self.env.poses_y[i])
            progress = np.linalg.norm(np.array(pos) - np.array(self._last_positions[i]))
            reward = float(progress * 10.0 + 0.1 - (100.0 if self.env.collisions[i] else 0.0))
            rewards.append(reward)
            self._last_positions[i] = pos

        obs_dict = self._convert_obs(obs)
        rew_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        terminated_dict = {agent: terminated for agent in self.agents}
        terminated_dict["__all__"] = terminated
        truncated_dict = {agent: truncated for agent in self.agents}
        truncated_dict["__all__"] = truncated

        return obs_dict, rew_dict, terminated_dict, truncated_dict, {agent: info for agent in self.agents}

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


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
        "reset_config": {"type": "rl_random_static"},
        "render_mode": render_mode,
    }


def setup_policies_and_config():
    """Setup policies and SAC configuration. Returns (policies, config)."""
    register_env("f1tenth_multi", lambda config: MultiAgentF110(config))

    # Create temporary environment for policy setup
    temp_env = MultiAgentF110(get_env_config())
    policies = {agent: PolicySpec(None, temp_env.observation_space[agent], temp_env.action_space[agent], {})
                for agent in temp_env.agents}
    temp_env.close()

    # Configure SAC with simplified replay buffer
    config = (SACConfig()
              .environment("f1tenth_multi", env_config=get_env_config())
              .framework("torch")
              .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
              .env_runners(num_env_runners=0)
              .training(
                  initial_alpha=1.0,
                  actor_lr=3e-4,
                  critic_lr=3e-4,
                  alpha_lr=3e-4,
                  lr=None,
                  target_entropy="auto",
                  n_step=1,  # Simplified to avoid buffer issues
                  tau=0.005,
                  train_batch_size_per_learner=256,
                  target_network_update_freq=1,
                  replay_buffer_config={
                      "type": "EpisodeReplayBuffer",  # Use simpler buffer
                      "capacity": 50000,
                  },
                  num_steps_sampled_before_learning_starts=1000,  # Reduced for faster start
    )
        .rl_module(
                  model_config=DefaultModelConfig(
                      fcnet_hiddens=[256, 256],
                      fcnet_activation="relu",
                  ),
    )
        .reporting(
                  metrics_num_episodes_for_smoothing=5,
                  min_sample_timesteps_per_iteration=1000,
    )
        .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id))

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
    log_dir = os.path.abspath(f"runs/multiagent_sac_{timestamp}")
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
    model_dir = os.path.abspath(f"models/multiagent_sac_run_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Setup policies and config
    policies, config = setup_policies_and_config()
    algo = setup_ray_and_algo(config)

    # Training loop
    TOTAL_TIMESTEPS = 20_000
    SAVE_EVERY = 1000  # Save less frequently to avoid issues

    try:
        while True:
            result = algo.train()

            # Use the correct key for timesteps
            timesteps_total = result.get('num_env_steps_sampled_lifetime', 0)
            print(f"Training iteration: {result.get('training_iteration', 0)}, Timesteps: {timesteps_total}")

            if timesteps_total > 0 and timesteps_total % SAVE_EVERY == 0:
                print(f"Saving model at {timesteps_total} timesteps")
                algo.save(model_dir)

            if timesteps_total >= TOTAL_TIMESTEPS:
                break

    except Exception as e:
        print(f"Training error: {e}")
        print("Attempting to save model before exit...")
        try:
            algo.save(model_dir)
        except:
            print("Could not save model")
        raise

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

    run_dirs = [d for d in os.listdir(models_dir) if d.startswith("multiagent_sac_run_")]
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
        obs_dict, _ = eval_env.reset(seed=42)
        done = False
        step_count = 0

        while not done and step_count < 1000:  # Reduced max steps for testing
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
