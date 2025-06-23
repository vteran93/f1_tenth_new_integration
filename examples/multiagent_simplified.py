import numpy as np
import ray
import gymnasium as gym
import gymnasium
import gymnasium.wrappers
import os
import time
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from f1tenth_gym.envs import F110Env
from enum import Enum

# Workaround for gymnasium compatibility issue with RLlib
import gymnasium.envs.registration
from enum import Enum

class VectorizeMode(Enum):
    ASYNC = "async"
    SYNC = "sync"

# Monkey patch the missing VectorizeMode
gymnasium.envs.registration.VectorizeMode = VectorizeMode

class SimpleMultiAgentF110(MultiAgentEnv):
    """Simple multi-agent wrapper for F110Env."""

    def __init__(self, env_config=None):
        config = env_config or {}
        self.env = F110Env(config=config, render_mode=config.get("render_mode"))
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        
        # Single agent action space
        single_action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 10.0]), dtype=np.float32
        )
        
        # Use the original observation space structure but for single agent
        # Extract single agent observation space from the multi-agent structure
        original_obs_space = self.env.observation_space.spaces
        single_obs_spaces = {}
        
        for key, space in original_obs_space.items():
            if key == 'ego_idx':
                single_obs_spaces[key] = gym.spaces.Box(
                    low=0, high=self.env.num_agents-1, shape=(), dtype=np.int32
                )
            elif key == 'scans':
                single_obs_spaces[key] = gym.spaces.Box(
                    low=space.low.min(), high=space.high.max(),
                    shape=(space.shape[1],), dtype=space.dtype
                )
            elif key == 'collisions':
                single_obs_spaces[key] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=np.float32  # Use float32 for consistency
                )
            elif key == 'lap_counts':
                single_obs_spaces[key] = gym.spaces.Box(
                    low=0, high=1000, shape=(), dtype=np.int32  # Reasonable upper bound for lap counts
                )
            else:
                # For other keys like poses_x, poses_y, etc., extract single agent dimension
                single_obs_spaces[key] = gym.spaces.Box(
                    low=space.low.min(), high=space.high.max(),
                    shape=(), dtype=space.dtype
                )
        
        single_obs_space = gym.spaces.Dict(single_obs_spaces)
        
        self.action_space = single_action_space
        self.observation_space = single_obs_space
        self._last_positions = [(0.0, 0.0)] * self.env.num_agents

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        
        # Split the multi-agent observation into per-agent observations
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            obs_dict[agent] = {
                'ego_idx': np.array(obs['ego_idx'], dtype=np.int32),
                'scans': np.clip(np.array(obs['scans'][i], dtype=np.float32), 0.0, 30.5),
                'poses_x': np.array(obs['poses_x'][i], dtype=np.float32),
                'poses_y': np.array(obs['poses_y'][i], dtype=np.float32), 
                'poses_theta': np.array(obs['poses_theta'][i], dtype=np.float32),
                'linear_vels_x': np.array(obs['linear_vels_x'][i], dtype=np.float32),
                'linear_vels_y': np.array(obs['linear_vels_y'][i], dtype=np.float32),
                'ang_vels_z': np.array(obs['ang_vels_z'][i], dtype=np.float32),
                'collisions': np.array(obs['collisions'][i], dtype=np.float32),
                'lap_times': np.array(obs['lap_times'][i], dtype=np.float32),
                'lap_counts': np.array(obs['lap_counts'][i], dtype=np.int32)
            }
        
        return obs_dict, {agent: info for agent in self.agents}

    def step(self, action_dict):
        actions = np.array([action_dict[agent] for agent in self.agents])
        obs, _, terminated, truncated, info = self.env.step(actions)
        done = terminated or truncated

        # Progress-based reward with collision penalty
        rewards = []
        for i in range(self.env.num_agents):
            current_pos = (self.env.poses_x[i], self.env.poses_y[i])
            last_pos = self._last_positions[i]
            
            # Calculate progress (simple euclidean distance moved)
            progress = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
            
            # Reward components:
            # 1. Progress reward (encourage forward movement)
            progress_reward = progress * 10.0  # Scale progress
            # 2. Collision penalty
            collision_penalty = 100.0 if self.env.collisions[i] else 0.0
            # 3. Small baseline reward for staying alive
            survival_reward = 0.1
            
            reward = progress_reward + survival_reward - collision_penalty
            rewards.append(reward)
            self._last_positions[i] = current_pos

        # Split observations back to per-agent format
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            obs_dict[agent] = {
                'ego_idx': np.array(obs['ego_idx'], dtype=np.int32),
                'scans': np.clip(np.array(obs['scans'][i], dtype=np.float32), 0.0, 30.5),
                'poses_x': np.array(obs['poses_x'][i], dtype=np.float32),
                'poses_y': np.array(obs['poses_y'][i], dtype=np.float32), 
                'poses_theta': np.array(obs['poses_theta'][i], dtype=np.float32),
                'linear_vels_x': np.array(obs['linear_vels_x'][i], dtype=np.float32),
                'linear_vels_y': np.array(obs['linear_vels_y'][i], dtype=np.float32),
                'ang_vels_z': np.array(obs['ang_vels_z'][i], dtype=np.float32),
                'collisions': np.array(obs['collisions'][i], dtype=np.float32),
                'lap_times': np.array(obs['lap_times'][i], dtype=np.float32),
                'lap_counts': np.array(obs['lap_counts'][i], dtype=np.int32)
            }
        
        rew_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        terminated_dict = {agent: terminated for agent in self.agents}
        terminated_dict["__all__"] = terminated
        truncated_dict = {agent: truncated for agent in self.agents}
        truncated_dict["__all__"] = truncated
        
        return obs_dict, rew_dict, terminated_dict, truncated_dict, {agent: info for agent in self.agents}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


# Toggle this to train or evaluate
train = False  # Set to True to train, False to evaluate

# Training configuration
TOTAL_TIMESTEPS = 20_000
SAVE_EVERY = 2000

if train:
    # Create directories
    timestamp = str(int(time.time()))
    run_id = f"multiagent_ppo_run_{timestamp}"
    model_dir = f"models/{run_id}"
    os.makedirs(model_dir, exist_ok=True)

    # Environment config
    env_config = {
        "map": "Spielberg",
        "num_agents": 2,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "original"},
        "reset_config": {"type": "rl_random_static"},
    }

    def env_creator(config):
        return SimpleMultiAgentF110(config)

    register_env("f1tenth_multi", env_creator)

    # Setup policies
    temp_env = env_creator(env_config)
    policies = {
        agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        for agent in temp_env.agents
    }

    # Configure PPO
    config = (
        PPOConfig()
        .environment("f1tenth_multi", env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=0)
        .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
    )

    ray.init(ignore_reinit_error=True)
    algo = config.build()
    
    # Train
    while True:
        result = algo.train()
        timesteps_total = result['timesteps_total']
        
        if timesteps_total % SAVE_EVERY == 0:  # Print and save every SAVE_EVERY timesteps
            print(f"Timesteps: {timesteps_total}")
            algo.save(model_dir)
        
        if timesteps_total >= TOTAL_TIMESTEPS:  # Stop at TOTAL_TIMESTEPS
            break
    
    # Save final model
    final_checkpoint = algo.save(model_dir)
    print(f"Model saved to {final_checkpoint}")
    algo.stop()

else:
    # Evaluation - find latest model
    models_dir = "models"
    run_dirs = [d for d in os.listdir(models_dir) if d.startswith("multiagent_ppo_run_")]
    
    if not run_dirs:
        print("No trained models found. Train a model first by setting train=True")
        exit(1)
    
    # Use newest model (sort by timestamp)
    latest_model = max(run_dirs, key=lambda x: int(x.split("_")[-1]))
    model_path = os.path.abspath(os.path.join(models_dir, latest_model))
    print(f"Using model: {latest_model}")
    print(f"Model path: {model_path}")
    
    # Config for temporary environment (NO rendering)
    temp_env_config = {
        "map": "Spielberg",
        "num_agents": 2,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "original"},
        "reset_config": {"type": "rl_random_static"},
        # NO render_mode here
    }
    
    # Config for evaluation environment (WITH rendering)
    eval_env_config = {
        "map": "Spielberg",
        "num_agents": 2,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "original"},
        "reset_config": {"type": "rl_random_static"},
        "render_mode": "human",
    }

    def env_creator(config):
        return SimpleMultiAgentF110(config)

    register_env("f1tenth_multi", env_creator)
    
    # Create temp env only for policy setup (no rendering), then close it
    temp_env = env_creator(temp_env_config)
    policies = {
        agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        for agent in temp_env.agents
    }
    temp_env.close()  # Close the temporary environment

    config = (
        PPOConfig()
        .environment("f1tenth_multi", env_config=temp_env_config)  # Change this line
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=0)
        .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
    )

    ray.init(ignore_reinit_error=True)
    algo = config.build()
    algo.restore(model_path)
    
    # Create evaluation environment only after algo is built
    eval_env = env_creator(eval_env_config)
    
    # Run evaluation episodes
    for episode in range(3):
        obs_dict, _ = eval_env.reset(seed=42)
        done = False
        step_count = 0
        
        while not done and step_count < 100000:
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                action = algo.compute_single_action(obs, policy_id=agent_id, explore=False)
                action_dict[agent_id] = action
            
            obs_dict, rew_dict, terminated_dict, truncated_dict, _ = eval_env.step(action_dict)
            eval_env.render()
            done = terminated_dict["__all__"] or truncated_dict["__all__"]
            step_count += 1
        
        print(f"Episode {episode + 1} completed in {step_count} steps")
    
    eval_env.close()
    algo.stop()

ray.shutdown()
