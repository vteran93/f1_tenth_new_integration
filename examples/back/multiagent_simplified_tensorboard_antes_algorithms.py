#!/usr/bin/env python3
"""
Simplified F1TENTH Multi-Agent Training/Evaluation Script
Supports both training and evaluation with configuration from a YAML file.
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
import yaml
import warnings
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from f1tenth_gym.envs import F110Env
from enum import Enum
from rewards import RewardStrategy, ProgressReward, SpeedTrackReward

# Fix for gymnasium compatibility with RLlib
import gymnasium.envs.registration

class VectorizeMode(Enum):
    ASYNC = "async"
    SYNC = "sync"

gymnasium.envs.registration.VectorizeMode = VectorizeMode


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # Validate mode
        mode = config.get("mode")
        if mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'.")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


class MultiAgentF110(MultiAgentEnv):
    """Simplified multi-agent wrapper for F110Env."""

    def __init__(self, env_config=None):
        self.env = F110Env(config=env_config or {}, render_mode=env_config.get("render_mode"))
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        self._last_positions = [(0.0, 0.0)] * self.env.num_agents
        
        # Initialize reward strategy
        reward_strategy_name = env_config.get("reward_strategy", "progress")
        reward_config = env_config.get("reward_config", {})
        if reward_strategy_name == "progress":
            self.reward_strategy = ProgressReward(self.env.num_agents, reward_config.get("progress", {}))
        elif reward_strategy_name == "speed_track":
            self.reward_strategy = SpeedTrackReward(self.env.num_agents, reward_config.get("speed_track", {}))
        else:
            raise ValueError(f"Unknown reward strategy: {reward_strategy_name}")
        
        # Extract single agent spaces from multi-agent F110Env
        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32), 
                                         high=np.array([1.0, 10.0], dtype=np.float32), 
                                         dtype=np.float32)
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
                                                      shape=(space.shape[1],), dtype=np.float32)
                else:
                    single_spaces[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(), 
                                                      shape=(), dtype=np.float32)
            else:
                single_spaces[key] = space
                
        return gym.spaces.Dict(single_spaces)

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
                        value[i].astype(np.float32),
                        original_space.low.min(),
                        original_space.high.max()
                    )
                else:
                    agent_obs[key] = np.array(value, dtype=np.float32)
            
            obs_dict[agent] = agent_obs
        return obs_dict

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        self.reward_strategy.reset(self._last_positions)
        return self._convert_obs(obs), {agent: info for agent in self.agents}

    def step(self, action_dict):
        actions = np.array([action_dict[agent] for agent in self.agents])
        obs, _, terminated, truncated, info = self.env.step(actions)

        # Calculate rewards using the reward strategy
        rewards = self.reward_strategy.compute_rewards(self.env)
        
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


def get_env_config(config, render_mode=None):
    """Get environment configuration from YAML config."""
    env_config = config.get("environment", {})
    env_config["render_mode"] = render_mode
    env_config["reward_strategy"] = config.get("reward", {}).get("strategy", "progress")
    env_config["reward_config"] = config.get("reward", {})
    return env_config


def setup_policies_and_config(config):
    """Setup policies and PPO configuration. Returns (policies, config)."""
    register_env("f1tenth_multi", lambda cfg: MultiAgentF110(cfg))
    
    # Create temporary environment for policy setup
    temp_env = MultiAgentF110(get_env_config(config))
    policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {}) 
                for agent in temp_env.agents}
    temp_env.close()
    
    # Configure PPO
    ppo_config = config.get("ppo", {})
    algo_config = (PPOConfig()
                   .environment("f1tenth_multi", env_config=get_env_config(config))
                   .framework("torch")
                   .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                   .env_runners(num_env_runners=0)
                   .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
                   .training(
                       lr=ppo_config.get("learning_rate", 0.0003),
                       gamma=ppo_config.get("gamma", 0.99),
                       train_batch_size=ppo_config.get("train_batch_size", 4000)
                   ))
    
    return policies, algo_config


def custom_log_creator(custom_path: str, prefix: str):
    """
    Returns a function __logger_creator(config) that will
    write TensorBoard event files to `custom_path`.
    """
    def logger_creator(config):
        os.makedirs(custom_path, exist_ok=True)
        run_dir = tempfile.mkdtemp(
            prefix=f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_",
            dir=custom_path
        )
        return UnifiedLogger(config, run_dir, loggers=None)
    return logger_creator


def setup_ray_and_algo(config, algo_config):
    """Initialize Ray and build algorithm—with TensorBoard logging."""
    ray.init(ignore_reinit_error=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(f"runs/multiagent_ppo_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    return algo_config.build(
        logger_creator=custom_log_creator(log_dir, "rllib_run")
    )


def setup_training(config):
    """Setup and run training."""
    print("Starting training...")
    
    # Setup
    timestamp = str(int(time.time()))
    model_dir = f"models/multiagent_ppo_run_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup policies and config
    policies, algo_config = setup_policies_and_config(config)
    algo = setup_ray_and_algo(config, algo_config)
    
    # Training loop
    training_config = config.get("training", {})
    total_timesteps = training_config.get("total_timesteps", 20000)
    save_every = training_config.get("save_every", 200)
    
    while True:
        result = algo.train()
        timesteps_total = result['timesteps_total']
        
        if timesteps_total % save_every == 0:
            print(f"Timesteps: {timesteps_total}")
            algo.save(model_dir)
            
        if timesteps_total >= total_timesteps:
            break
    
    final_checkpoint = algo.save(model_dir)
    print(f"Training completed. Model saved to {final_checkpoint}")
    algo.stop()


def setup_evaluation(config):
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
    policies, algo_config = setup_policies_and_config(config)
    algo = setup_ray_and_algo(config, algo_config)
    algo.restore(model_path)
    
    # Run evaluation
    evaluation_config = config.get("evaluation", {})
    num_episodes = evaluation_config.get("num_episodes", 3)
    max_steps = evaluation_config.get("max_steps_per_episode", 100000)
    seed = evaluation_config.get("seed", 42)
    
    eval_env = MultiAgentF110(get_env_config(config, render_mode="human"))
    
    for episode in range(num_episodes):
        obs_dict, _ = eval_env.reset(seed=seed)
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
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
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)  # Hidden argument for backward compatibility
    args = parser.parse_args()
    
    if args.train:
        warnings.warn("The --train argument is deprecated. Use 'mode: train' or 'mode: eval' in the config.yaml file instead.")
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute based on mode
    mode = config.get("mode")
    if mode == "train":
        setup_training(config)
    else:  # mode == "eval"
        setup_evaluation(config)
    
    ray.shutdown()