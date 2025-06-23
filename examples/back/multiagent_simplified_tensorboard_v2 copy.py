
#!/usr/bin/env python3
"""
Simplified F1TENTH-Agent Training/Evaluation Script
Supports both training and evaluation with configuration from YAML file.
"""

import numpy as np
import ray
import gymnasium as gym
import tempfile
from datetime import datetime
import os
import time
import warnings
import json
import csv
import argparse
import yaml
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from f1tenth_gym import envs as F110Env
from enum import Enum
from rewards import RewardStrategy, ProgressReward, SpeedTrackReward, CrossTrackHeadReward, TALearningReward, RacePerformanceReward
from algorithms import get_algorithm
from ray.tune.logger import LoggerCallback
from tensorboardX import SummaryWriter

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
        # Validate agent configurations
        num_agents = config.get("environment", {}).get("num_agents", 1)
        reward_agents = config.get("reward", {}).get("agents", {})
        algo_agents = config.get("algorithm", {}).get("agents", {})
        for agent_id in [f"agent_{i}" for i in range(num_agents)]:
            if agent_id not in reward_agents:
                raise ValueError(f"Reward configuration missing for {agent_id}")
            if agent_id not in algo_agents:
                raise ValueError(f"Algorithm configuration missing for {agent_id}")
            # Validate reward strategy
            strategy = reward_agents[agent_id].get("strategy")
            if strategy not in ["progress", "speed_track", "crosstrack", "talearning", "raceperformance"]:
                raise ValueError(f"Invalid reward strategy for {agent_id}: {strategy}")
            # Validate algorithm type
            algo_type = algo_agents[agent_id].get("type")
            if algo_type not in ["ppo", "sac", "ddpg", "td3"]:
                raise ValueError(f"Invalid algorithm type for {agent_id}: {algo_type}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

class MultiAgentF110(MultiAgentEnv):
    """Simplified multi-agent wrapper for F110Env."""

    def __init__(self, env_config=None):
        super().__init__()  # Initialize parent MultiAgentEnv
        if env_config is None:
            env_config = {}
        if "reward_config" not in env_config or "agents" not in env_config.get("reward_config", {}):
            raise ValueError(f"Invalid env_config: 'reward_config' or 'reward_config.agents' missing. Got env_config={env_config}")
        print(f"MultiAgentF110: Initializing with env_config: {env_config}")
        
        #self.env = F110Env(config=env_config or {}, render_mode=env_config.get("render_mode"))
        self.env = F110Env.F110Env(config=env_config or {}, render_mode=env_config.get("render_mode"))
        #self.env = F110Env(config=env_config or {}, render_mode=env_config.get("render_mode"))
        
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        self._last_positions = [(0.0, 0.0)] * self.env.num_agents
        self._agent_ids = set(self.agents)  # Explicitly set for MultiAgentEnv
        
        # Initialize per-agent reward strategies
        reward_config = env_config.get("reward_config", {})
        self.reward_strategies = {}
        for agent_id in self.agents:
            strategy_name = reward_config["agents"][agent_id].get("strategy")
            strategy_config = reward_config["agents"][agent_id].get(strategy_name, {})
            if strategy_name == "progress":
                self.reward_strategies[agent_id] = ProgressReward(self.env.num_agents, strategy_config)
            elif strategy_name == "speed_track":
                self.reward_strategies[agent_id] = SpeedTrackReward(self.env.num_agents, strategy_config)
            elif strategy_name == "crosstrack":
                self.reward_strategies[agent_id] = CrossTrackHeadReward(self.env.num_agents, env_config)
            elif strategy_name == "talearning":
                self.reward_strategies[agent_id] = TALearningReward(self.env.num_agents, strategy_config)
            elif strategy_name == "raceperformance":
                self.reward_strategies[agent_id] = RacePerformanceReward(self.env.num_agents, strategy_config)
            else:
                raise ValueError(f"Unknown reward strategy for {agent_id}: {strategy_name}")

        # Define per-agent action and observation spaces
        single_action_space = gym.spaces.Box(low=np.array([1.0, -0.5], dtype=np.float32), 
                                            high=np.array([10.0, 0.5], dtype=np.float32), 
                                            dtype=np.float32)
        single_observation_space = self._make_single_agent_obs_space()
        self.action_spaces = {agent: single_action_space for agent in self.agents}
        self.observation_spaces = {agent: single_observation_space for agent in self.agents}
        self._spaces_in_preferred_format = True  # Indicate spaces are in dict format

    def _make_single_agent_obs_space(self):
        """Create single agent observation space from F110Env's multi-agent space."""
        orig_spaces = self.env.observation_space.spaces
        single_spaces = {}
        
        for key, space in orig_spaces.items():
            if key == 'ego_idx':
                single_spaces[key] = gym.spaces.Box(low=0, high=space.n-1, shape=(), dtype=np.int64)
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
                    agent_obs[key] = np.array(value, dtype=np.int64)
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
        for agent_id in self.agents:
            self.reward_strategies[agent_id].reset(self._last_positions)
        return self._convert_obs(obs), {agent: info for agent in self.agents}

    def step(self, action_dict):
        actions = np.array([action_dict[agent] for agent in self.agents])
        # Update actions for relevant reward strategies
        for agent_id in self.agents:
            if isinstance(self.reward_strategies[agent_id], (TALearningReward, RacePerformanceReward)):
                self.reward_strategies[agent_id].update_actions(actions)
        
        obs, _, terminated, truncated, info = self.env.step(actions)

        # Calculate rewards per agent
        rew_dict = {}
        for i, agent_id in enumerate(self.agents):
            rewards = self.reward_strategies[agent_id].compute_rewards(self.env, obs)
            rew_dict[agent_id] = rewards[i]

        obs_dict = self._convert_obs(obs)
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
    env_config["reward_config"] = {"agents": config.get("reward", {}).get("agents", {})}
    print(f"get_env_config: env_config={env_config}")
    return env_config

def setup_policies_and_config(config):
    """Setup policies and per-agent algorithms. Returns (policies, algorithms)."""
    env_name = "f1tenth_multi"
    env_config = get_env_config(config)
    register_env(env_name, lambda cfg: MultiAgentF110({**env_config, **cfg}))
    
    # Create temporary environment for policy setup
    print(f"setup_policies_and_config: Creating temp_env with env_config={env_config}")
    temp_env = MultiAgentF110(env_config)
    policies = {agent: PolicySpec(None, temp_env.observation_spaces[agent], temp_env.action_spaces[agent], {}) 
                for agent in temp_env.agents}
    temp_env.close()
    
    # Create algorithm instances for each agent
    algorithms = {}
    for agent_id in temp_env.agents:
        algo_config = config.get("algorithm", {}).get("agents", {}).get(agent_id, {})
        algorithms[agent_id] = get_algorithm(algo_config, env_name, policies, agent_id, env_config)
        algorithms[agent_id].setup_config()
    
    return policies, algorithms

def filter_serializable(data, max_depth=5, current_depth=0):
    """Recursively filter a dictionary to keep only JSON-serializable values."""
    if current_depth > max_depth:
        return None  # Prevent infinite recursion
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, (list, tuple)):
        return [filter_serializable(item, max_depth, current_depth + 1) for item in data]
    elif isinstance(data, dict):
        return {k: filter_serializable(v, max_depth, current_depth + 1) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, (np.int32, np.int64, np.float32, np.float64)):
        return data.item()  # Convert numpy scalars to Python scalars
    else:
        return str(data)  # Convert non-serializable objects to strings

def custom_log_creator(custom_path: str, prefix: str):
    """
    Creates a logger creator function that produces a logger object compatible with RLlib,
    using custom JSON/CSV logging and TensorBoard logging via SummaryWriter.
    """
    def logger_creator(config):
        os.makedirs(custom_path, exist_ok=True)
        log_dir = tempfile.mkdtemp(
            prefix=f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_",
            dir=custom_path
        )

        class CustomLogger(LoggerCallback):
            def __init__(self):
                super().__init__()
                self.logdir = log_dir
                self.iteration = 0
                self.json_file = os.path.join(log_dir, "result.json")
                self.csv_file = os.path.join(log_dir, "result.csv")
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                self.csv_fields = None

            def on_result(self, result):
                self.iteration += 1
                # Filter result to ensure JSON-serializable
                filtered_result = filter_serializable(result)
                filtered_result["iteration"] = self.iteration
                # Log to JSON
                with open(self.json_file, 'a') as f:
                    json.dump(filtered_result, f)
                    f.write('\n')
                # Log to CSV
                if self.csv_fields is None:
                    self.csv_fields = list(filtered_result.keys())
                    with open(self.csv_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                        writer.writeheader()
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                    writer.writerow(filtered_result)
                # Log to TensorBoard
                for key, value in filtered_result.items():
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(key, value, self.iteration)
                self.tb_writer.flush()

            def close(self):
                self.tb_writer.close()

        return CustomLogger()

    return logger_creator

def setup_ray_and_algo(config, algorithms):
    """Initialize Ray and build algorithms with TensorBoard logging."""
    ray.init(ignore_reinit_error=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(f"runs/multiagent_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    for agent_id, algo in algorithms.items():
        algo.build(logger_creator=custom_log_creator(log_dir, f"rllib_{agent_id}_{config['algorithm']['agents'][agent_id]['type']}"))
    return algorithms

def setup_training(config):
    """Setup and run training with best model saving."""
    print(f"Starting training with per-agent algorithms...")
    
    # Setup
    timestamp = str(int(time.time()))
    model_dir = f"models/multiagent_run_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    best_model_dir = os.path.join(model_dir, "best")
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Setup policies and algorithms
    policies, algorithms = setup_policies_and_config(config)
    algorithms = setup_ray_and_algo(config, algorithms)
    
    # Training loop
    training_config = config.get("training", {})
    total_timesteps = training_config.get("total_timesteps", 20000)
    save_every = training_config.get("save_every", 200)
    best_metric = training_config.get("save_best_metric", "episode_reward_mean")
    best_metric_value = training_config.get("save_best_threshold", float('-inf'))
    
    while True:
        results = {}
        for agent_id, algo in algorithms.items():
            results[agent_id] = algo.train()
        
        timesteps_total = results[list(algorithms.keys())[0]]['timesteps_total']
        
        # Calculate average episode reward across agents
        avg_episode_reward = np.mean([results[agent_id].get(best_metric, 0.0) for agent_id in algorithms])
        
        if timesteps_total % save_every == 0:
            print(f"Timesteps: {timesteps_total}, Avg Episode Reward: {avg_episode_reward}")
            for agent_id, algo in algorithms.items():
                algo.save(os.path.join(model_dir, f"checkpoint_{agent_id}_{timesteps_total}"))
            
            # Save best model if metric improves
            if avg_episode_reward > best_metric_value:
                best_metric_value = avg_episode_reward
                print(f"New best model found with {best_metric}: {best_metric_value}")
                for agent_id, algo in algorithms.items():
                    algo.save(os.path.join(best_model_dir, f"best_{agent_id}"))
        
        if timesteps_total >= total_timesteps:
            break
    
    print(f"Training completed. Final checkpoints saved to {model_dir}")
    print(f"Best model saved to {best_model_dir} with {best_metric}: {best_metric_value}")
    for algo in algorithms.values():
        algo.stop()

def setup_evaluation(config):
    """Setup and run evaluation with best model."""
    print(f"Starting evaluation with per-agent algorithms...")
    
    # Find latest model
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found. Train a model first.")
        return
        
    run_dirs = [d for d in os.listdir(models_dir) if d.startswith("multiagent_run_")]
    if not run_dirs:
        print("No trained models found. Train a model first.")
        return
    
    latest_run = max(run_dirs, key=lambda x: int(x.split("_")[-1]))
    best_model_path = os.path.join(models_dir, latest_run, "best")
    if not os.path.exists(best_model_path):
        print(f"No best model found in {best_model_path}. Train a model first.")
        return
    
    print(f"Using best model from: {best_model_path}")
    
    # Setup algorithms
    policies, algorithms = setup_policies_and_config(config)
    algorithms = setup_ray_and_algo(config, algorithms)
    for agent_id, algo in algorithms.items():
        algo.restore(os.path.join(best_model_path, f"best_{agent_id}"))
    
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
        episode_rewards = {agent_id: 0.0 for agent_id in algorithms}
        
        while not done and step_count < max_steps:
            action_dict = {agent_id: algo.compute_single_action(obs, policy_id=agent_id, explore=False)
                          for agent_id, algo in algorithms.items() for obs in [obs_dict[agent_id]]}
            
            obs_dict, rew_dict, terminated_dict, truncated_dict, _ = eval_env.step(action_dict)
            eval_env.render()
            done = terminated_dict["__all__"] or truncated_dict["__all__"]
            step_count += 1
            for agent_id in episode_rewards:
                episode_rewards[agent_id] += rew_dict[agent_id]
        
        print(f"Episode {episode + 1} completed in {step_count} steps")
        print(f"Episode rewards: {episode_rewards}")
    
    eval_env.close()
    for algo in algorithms.values():
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