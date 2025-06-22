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
        self._crashed_agents = set()  # Track which agents have crashed
        
        # Extract single agent spaces from multi-agent F110Env
        self.action_space = self._make_single_agent_action_space()
        self.observation_space = self._make_single_agent_obs_space()

    def _make_single_agent_obs_space(self):
        """Create single agent observation space from F110Env's multi-agent space."""
        orig_spaces = self.env.observation_space.spaces
        single_spaces = {}
        
        for key, space in orig_spaces.items():
            if key == 'ego_idx':
                # ego_idx is Discrete, convert to Box for single agent
                single_spaces[key] = gym.spaces.Box(low=0, high=space.n-1, shape=(), dtype=np.int32)
            elif hasattr(space, 'shape') and len(space.shape) > 0 and space.shape[0] == self.env.num_agents:
                # Multi-agent dimension - extract single agent space
                if key == 'scans':
                    # scans: (num_agents, num_beams) -> (num_beams,)
                    single_spaces[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(), 
                                                      shape=(space.shape[1],), dtype=space.dtype)
                else:
                    # Other multi-agent arrays: (num_agents,) -> ()
                    single_spaces[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(), 
                                                      shape=(), dtype=space.dtype)
            else:
                # Keep space as-is (shouldn't happen in current F110Env)
                single_spaces[key] = space
                
        return gym.spaces.Dict(single_spaces)

    def _convert_obs(self, obs):
        """Convert multi-agent observation to per-agent format."""
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            agent_obs = {}
            for key, value in obs.items():
                if key == 'ego_idx':
                    # ego_idx is the same for all agents
                    agent_obs[key] = np.array(value, dtype=np.int32)
                elif hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == self.env.num_agents:
                    # Multi-agent observation - extract for this agent
                    original_space = self.env.observation_space.spaces[key]
                    agent_obs[key] = np.clip(
                        value[i].astype(original_space.dtype),
                        original_space.low.min(),
                        original_space.high.max()
                    )
                else:
                    # Single value for all agents
                    agent_obs[key] = np.array(value, dtype=value.dtype if hasattr(value, 'dtype') else np.float32)
            
            obs_dict[agent] = agent_obs
        return obs_dict

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        self._crashed_agents = set()  # Reset crashed agents
        return self._convert_obs(obs), {agent: info for agent in self.agents}

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
        info_dict = {agent: info for agent in obs_dict.keys()}
        
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
        "reset_config": {"type": "rl_random_static"},
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
              .training(train_batch_size=4000)
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
