#!/usr/bin/env python3
"""
Simplified F1TENTH Multi-Agent Training/Evaluation Script
Supports both training and evaluation with minimal configuration.
"""

import numpy as np
import ray
from ray.tune.logger import UnifiedLogger         # (1)
import gymnasium as gym
from gymnasium.spaces.utils import flatten, flatten_space
import tempfile
from datetime import datetime
import os
import time
import argparse
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from f1tenth_gym.envs import F110Env
from enum import Enum
from ray.rllib.connectors.env_to_module import EnvToModulePipeline
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
import torch
# + Añadir para extraer acciones si insistimos en usar Columns:
# from ray.rllib.core.columns import Columns

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
        
        # Extract single agent spaces from multi-agent F110Env
        self.action_space = self._make_single_agent_action_space()
        # keep original for flattening...
        orig_space = self._make_single_agent_obs_space()
        self._orig_obs_space = orig_space
        # ... but expose the flat Box to RLlib
        self.observation_space = flatten_space(orig_space)


        # Multi-agent dicts (keyed by agent ID)
        self.action_spaces = {
            agent: self.action_space for agent in self.agents
        }
        self.observation_spaces = {
            agent: self.observation_space for agent in self.agents
        }

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
            # 1) build the same dict you had before
            raw = {}
            for key, value in obs.items():
                if key == 'ego_idx':
                    # ego_idx is the same for all agents
                    raw[key] = np.array(value, dtype=np.int32)
                elif hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == self.env.num_agents:
                    # Multi-agent observation - extract for this agent
                    original_space = self.env.observation_space.spaces[key]
                    raw[key] = np.clip(
                        value[i].astype(original_space.dtype),
                        original_space.low.min(),
                        original_space.high.max()
                    )
                else:
                    # Single value for all agents
                    raw[key] = np.array(value, dtype=value.dtype if hasattr(value, 'dtype') else np.float32)
        
            # 2) flatten that dict → 1-D array
            flat = flatten(self._orig_obs_space, raw)
            obs_dict[agent] = flat
        return obs_dict

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        return self._convert_obs(obs), {agent: info for agent in self.agents}

    def step(self, action_dict):
        actions = np.stack([action_dict[agent] for agent in self.agents]).astype(self.env.action_space.dtype)
        obs, _, terminated, truncated, info = self.env.step(actions)

        # Improved reward function that encourages forward movement
        rewards = []
        for i in range(self.env.num_agents):
            # Reward components:
            reward = 0.0
            
            # 1. Speed reward - encourage positive forward speed
            speed = action_dict[f"agent_{i}"][0]  # First action is speed
            if speed > 0:
                reward += speed * 5.0  # Reward forward speed
            else:
                reward += speed * 20.0  # Heavy penalty for backward speed
            
            # 2. Steering penalty - encourage smooth steering
            steering = abs(action_dict[f"agent_{i}"][1])  # Second action is steering
            reward -= steering * 2.0  # Penalty for sharp turns
            
            # 3. Collision penalty
            if self.env.collisions[i]:
                reward -= 200.0
            
            # 4. Track position reward - stay on track
            # Use distance to track centerline (if available)
            # For now, just give a small survival bonus
            reward += 1.0  # Survival bonus
            
            # 5. Lap progress reward (if available)
            if hasattr(self.env, 'lap_times') and self.env.lap_times[i] > 0:
                reward += 50.0  # Lap completion bonus
            
            rewards.append(reward)
            self._last_positions[i] = (self.env.poses_x[i], self.env.poses_y[i])

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


def setup_policies_and_config(render_mode=None):
    # 1) Registrar el entorno
    register_env(
        "f1tenth_multi",
        lambda cfg: MultiAgentF110(cfg)
    )

    # 2) Instanciar wrapper para leer agentes y espacios
    wrapped = MultiAgentF110(get_env_config(render_mode=render_mode))
    agents = wrapped.agents
    act_space = wrapped.action_space
    flat_obs_space = wrapped.observation_space
    wrapped.close()

    # 3) Construir PolicySpecs
    policies = {
        agent: PolicySpec(
            None,
            observation_space=flat_obs_space,
            action_space=act_space,
            config={}
        )
        for agent in agents
    }

    # 4) Configurar PPO
    config = (
        PPOConfig()
        .environment("f1tenth_multi", env_config=get_env_config(render_mode=render_mode))
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)
        .env_runners()  # usa default V2
        .multi_agent(policies=policies,
                     policy_mapping_fn=lambda aid, *_: aid)
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
    ray.init(ignore_reinit_error=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(f"runs/multiagent_ppo_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    return config.build_algo(
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
    TOTAL_TIMESTEPS = 5_000  # Reduced for faster testing
    SAVE_EVERY = 100
    
    while True:
        result = algo.train()
        
        # Use the correct key for timesteps under the new API stack
        # According to RLlib docs, "num_env_steps_sampled_lifetime" is the key for global timesteps
        timesteps_total = result.get("num_env_steps_sampled_lifetime", 0)
        
        if timesteps_total % SAVE_EVERY == 0:
            print(f"Timesteps: {timesteps_total}")
            # Use absolute path for saving to avoid PyArrow URI issues
            algo.save(os.path.abspath(model_dir))
            
        if timesteps_total >= TOTAL_TIMESTEPS:
            break
    
    final_checkpoint = algo.save(os.path.abspath(model_dir))
    print(f"Training completed. Model saved to {final_checkpoint}")
    algo.stop()


def setup_evaluation():
    print("Starting evaluation...")
    # 1) localizar la carpeta del último checkpoint
    models_dir = "models"
    run_dirs = [d for d in os.listdir(models_dir) if d.startswith("multiagent_ppo_run_")]
    latest = max(run_dirs, key=lambda x: int(x.split("_")[-1]))
    ckpt_dir = os.path.abspath(os.path.join(models_dir, latest))
    print(f"Using checkpoint dir: {ckpt_dir}\n")

    # 2) reconstruir la configuración con modo evaluación (New API Stack)
    policies, config = setup_policies_and_config(render_mode="human")
    ray.init(ignore_reinit_error=True)
    algo = config.build_algo(
        logger_creator=custom_log_creator("runs/eval", "eval_run")
    )
    algo.restore(ckpt_dir)

    # 3) Crear env manualmente para renderizado
    env = MultiAgentF110(get_env_config(render_mode="human"))

    # 4) Obtener RLModules para cada agente
    modules = {agent: algo.get_module(agent) for agent in env.agents}

    for ep in range(3):  # 3 episodios
        obs, _ = env.reset()
        done = {agent: False for agent in env.agents}
        done["__all__"] = False
        while not done["__all__"]:
            actions = {}
            for agent in env.agents:
                # RLModule expects a dict with 'obs' key and a torch tensor batch
                obs_batch = {"obs": torch.from_numpy(np.expand_dims(obs[agent], axis=0)).float()}
                module = modules[agent]
                out = module.forward_inference(obs_batch)
                logits = out["action_dist_inputs"]
                action_dist_class = module.get_inference_action_dist_cls()
                action_dist = action_dist_class.from_logits(logits)
                action = action_dist.sample()
                # Clip action to env bounds
                unclipped = action[0].cpu().numpy()
                clipped = np.clip(unclipped, env.action_space.low, env.action_space.high)
                print(f"{agent} action (unclipped): {unclipped}, clipped: {clipped}")
                actions[agent] = clipped
            obs, rewards, done, truncated, info = env.step(actions)
            env.render()
        print(f"Episode {ep+1} finished.")
    env.close()

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1TENTH Multi-Agent RL Training/Evaluation")
    parser.add_argument("--train", action="store_true", help="Run training mode instead of evaluation")
    args = parser.parse_args()
    
    if args.train:
        setup_training()
    else:
        setup_evaluation()
    
    ray.shutdown()
