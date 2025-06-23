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
from f1tenth_gym.envs import F110Env
from enum import Enum
from rewards import RewardStrategy, ProgressReward, SpeedTrackReward, CrossTrackHeadReward, TALearningReward, RacePerformanceReward
from algorithms import get_algorithm

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
        # Validate algorithm type
        algo_type = config.get("algorithm", {}).get("type")
        if algo_type not in ["ppo", "sac", "ddpg", "td3"]:
            raise ValueError(f"Invalid algorithm type: {algo_type}. Supported: ['ppo', 'sac', 'ddpg', 'td3']")
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
        
        # Initialize reward strategies
        reward_strategy_name = env_config.get("reward_strategy", "progress")
        reward_config = env_config.get("reward_config", {})
        self.reward_strategies = []
        
        if reward_strategy_name == "combined":
            for strategy_name in ["progress", "speed_track", "crosstrack", "talearning", "raceperformance"]:
                if strategy_name in reward_config:
                    if strategy_name == "progress":
                        self.reward_strategies.append(ProgressReward(self.env.num_agents, reward_config["progress"]))
                    elif strategy_name == "speed_track":
                        self.reward_strategies.append(SpeedTrackReward(self.env.num_agents, reward_config["speed_track"]))
                    elif strategy_name == "crosstrack":
                        self.reward_strategies.append(CrossTrackHeadReward(self.env.num_agents, reward_config["crosstrack"]))
                    elif strategy_name == "talearning":
                        self.reward_strategies.append(TALearningReward(self.env.num_agents, reward_config["talearning"]))
                    elif strategy_name == "raceperformance":
                        self.reward_strategies.append(RacePerformanceReward(self.env.num_agents, reward_config["raceperformance"]))
        else:
            if reward_strategy_name == "progress":
                self.reward_strategies.append(ProgressReward(self.env.num_agents, reward_config.get("progress", {})))
            elif reward_strategy_name == "speed_track":
                self.reward_strategies.append(SpeedTrackReward(self.env.num_agents, reward_config.get("speed_track", {})))
            elif reward_strategy_name == "crosstrack":
                self.reward_strategies.append(CrossTrackHeadReward(self.env.num_agents, reward_config.get("crosstrack", {})))
            elif reward_strategy_name == "talearning":
                self.reward_strategies.append(TALearningReward(self.env.num_agents, reward_config.get("talearning", {})))
            elif reward_strategy_name == "raceperformance":
                self.reward_strategies.append(RacePerformanceReward(self.env.num_agents, reward_config.get("raceperformance", {})))
            else:
                raise ValueError(f"Unknown reward strategy: {reward_strategy_name}")
        
        # Extract single agent spaces from multi-agent F110Env
        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32), 
                                         high=np.array([1.0, 10.0], dtype=np.float32), 
                                         dtype=np.float32)
        self.observation_space = self._make_single_agent_obs_space()
        
        # For tracking reward components
        self.episode_reward_components = {}

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
                        value[i].numpy() if hasattr(value, 'numpy') else value[i].astype(np.float32),
                        original_space.low.min(),
                        original_space.high.max()
                    )
                else:
                    agent_obs[key] = np.array(value, dtype=np.float32)
            
            obs_dict[agent] = agent_obs
        return obs_dict

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        for strategy in self.reward_strategies:
            strategy.reset(self._last_positions)
        return self._convert_obs(obs), {agent: info for agent in self.agents}

    def step(self, action_dict):
        actions = np.array([action_dict[agent] for agent in self.agents])
        
        # Pasar las acciones a todas las estrategias de recompensa
        for strategy in self.reward_strategies:
            if hasattr(strategy, 'update_actions'):
                strategy.update_actions(actions)
            elif isinstance(strategy, RacePerformanceReward):
                strategy.actions = actions
        
        obs, _, terminated, truncated, info = self.env.step(actions)

        # Calculate rewards using all strategies
        rewards = np.zeros(self.env.num_agents)
        for strategy in self.reward_strategies:
            try:
                strategy_rewards = np.array(strategy.compute_rewards(self.env, obs))
                rewards += strategy_rewards
                
                # Track reward components for logging (cambia env por self.env)
                if isinstance(strategy, RacePerformanceReward):
                    self.episode_reward_components = {
                        'progress_reward': strategy.progress_scale * np.mean([np.sqrt((self.env.poses_x[i] - self._last_positions[i][0])**2 + 
                                                                        (self.env.poses_y[i] - self._last_positions[i][1])**2) 
                                        for i in range(self.env.num_agents)]),
                        'speed_reward': strategy.speed_scale * np.mean(obs["linear_vels_x"]),
                        'collision_penalty': -strategy.collision_penalty * sum(self.env.collisions),
                        'overtake_bonus': np.mean([getattr(strategy, 'overtake_bonuses', [0]*self.env.num_agents)]),
                        'stall_penalty': -strategy.stall_penalty * sum([s < strategy.min_speed_threshold for s in obs["linear_vels_x"]]),
                        'jerk_penalty': -strategy.jerk_penalty * np.mean(np.abs(np.diff([a[0] for a in actions])))
                    }
            except Exception as e:
                print(f"Error computing rewards with {strategy.__class__.__name__}: {str(e)}")
                raise

        obs_dict = self._convert_obs(obs)
        rew_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        terminated_dict = {agent: terminated for agent in self.agents}
        terminated_dict["__all__"] = terminated
        truncated_dict = {agent: truncated for agent in self.agents}
        truncated_dict["__all__"] = truncated
        
        # Add reward components to info for logging
        for agent in self.agents:
            info[agent].update(self.episode_reward_components)
        
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
    """Setup policies and algorithm configuration. Returns (policies, algorithm)."""
    env_name = "f1tenth_multi"
    register_env(env_name, lambda cfg: MultiAgentF110(cfg))
    
    # Create temporary environment for policy setup
    env_config = get_env_config(config)
    temp_env = MultiAgentF110(env_config)
    policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {}) 
                for agent in temp_env.agents}
    temp_env.close()
    
    # Create algorithm instance, passing full config
    algo = get_algorithm(config, env_name, policies)
    algo.setup_config()
    
    return policies, algo

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

def setup_ray_and_algo(config, algo):
    """Initialize Ray and build algorithm with TensorBoard logging."""
    ray.init(ignore_reinit_error=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(f"runs/multiagent_{config['algorithm']['type']}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    algo.build(logger_creator=custom_log_creator(log_dir, f"rllib_{config['algorithm']['type']}"))
    return algo

def setup_training(config):
    """Setup and run training."""
    print(f"Starting training with algorithm: {config['algorithm']['type']}...")
    
    # Setup
    timestamp = str(int(time.time()))
    model_dir = f"models/multiagent_{config['algorithm']['type']}_run_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup policies and algorithm
    policies, algo = setup_policies_and_config(config)
    algo = setup_ray_and_algo(config, algo)
    
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
    print(f"Starting evaluation with algorithm: {config['algorithm']['type']}...")
    
    # Find latest model
    models_dir = "models"
    algo_type = config['algorithm']['type']
    if not os.path.exists(models_dir):
        print("No models directory found. Train a model first.")
        return
        
    run_dirs = [d for d in os.listdir(models_dir) if d.startswith(f"multiagent_{algo_type}_run_")]
    if not run_dirs:
        print(f"No trained models found for algorithm {algo_type}. Train a model first.")
        return
    
    latest_model = max(run_dirs, key=lambda x: int(x.split("_")[-1]))
    model_path = os.path.abspath(os.path.join(models_dir, latest_model))
    print(f"Using model: {latest_model}")
    
    # Setup algorithm
    policies, algo = setup_policies_and_config(config)
    algo = setup_ray_and_algo(config, algo)
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