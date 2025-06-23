import os
import time
from typing import Dict
import yaml
import numpy as np
import torch
import ray
from ray.rllib.algorithms import ppo, sac
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from f1tenth_gym.envs import F110Env
from rewards import RacePerformanceReward, CrossTrackHeadReward
import gymnasium as gym

class MultiAgentF110(MultiAgentEnv):
    def __init__(self, config: Dict):
        # Almacenar config como atributo de instancia
        self.config = config
        env_config = config.get("environment", {})
        self._agents = [f"agent_{i}" for i in range(env_config.get("num_agents", 2))]
        self._possible_agents = self._agents
        self._observation_space = None  # Inicializar como None, se definirá después
        self._action_space = None      # Inicializar como None, se definirá después
        
        super().__init__()  # Llamar a super().__init__ después de inicializar agentes

        print(f"MultiAgentF110: Initializing with config: {config}")
        reward_config = config.get("reward", {})
        self.render_training = config["training"].get("render_training", False)
        self.render_interval = config["training"].get("render_interval", 200)
        self.step_count = 0
        self.episode_step_count = 0
        # Forzar render_mode=None inicialmente y controlarlo manualmente
        self.env = F110Env(config=env_config or {}, render_mode=None)
        self.num_beams = env_config.get("num_beams", 5)  # Valor por defecto
        self._observation_space = {
            agent_id: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_beams + 3,), dtype=np.float32)
            for agent_id in self._agents
        }
        self._action_space = {
            agent_id: gym.spaces.Box(low=np.array([0.0, -0.5]), high=np.array([10.0, 0.5]), dtype=np.float32)
            for agent_id in self._agents
        }
        self.reward_fn = {}
        for agent_id in self._agents:
            strategy = reward_config.get("agents", {}).get(agent_id, {}).get("strategy")
            reward_params = reward_config.get("agents", {}).get(agent_id, {}).get(strategy, {})
            if strategy == "raceperformance":
                self.reward_fn[agent_id] = RacePerformanceReward(**reward_params)
            elif strategy == "crosstrack":
                self.reward_fn[agent_id] = CrossTrackHeadReward(**reward_params)
        self.last_components = {agent_id: {} for agent_id in self._agents}
        self.episode_rewards = {agent_id: 0.0 for agent_id in self._agents}
        print(f"Action space: {self._action_space}")
        print(f"Observation space: {self._observation_space}")
        print(f"Render training: {self.render_training}, Render interval: {self.render_interval}, Mode: {config.get('mode')}")

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, value):
        self._agents = value

    @property
    def possible_agents(self):
        return self._possible_agents

    @possible_agents.setter
    def possible_agents(self, value):
        self._possible_agents = value

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def get_action_space(self, agent_id):
        return self._action_space[agent_id]

    def get_observation_space(self, agent_id):
        return self._observation_space[agent_id]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        print(f"Reset obs: poses_x={obs['poses_x']}, poses_y={obs['poses_y']}, poses_theta={obs['poses_theta']}")
        print(f"Reset info: {info}")
        self.episode_rewards = {agent_id: 0.0 for agent_id in self._agents}
        self.step_count = 0
        self.episode_step_count = 0
        obs_dict = {}
        info_dict = {agent_id: info for agent_id in self._agents}
        for i, agent_id in enumerate(self._agents):
            agent_obs = np.concatenate([
                obs['scans'][i],
                np.array([
                    obs['poses_x'][i],
                    obs['poses_y'][i],
                    obs['poses_theta'][i]
                ])
            ])
            obs_dict[agent_id] = agent_obs.astype(np.float32)
        for agent_id in self._agents:
            positions = [(obs['poses_x'][i], obs['poses_y'][i]) for i in range(len(self._agents))]
            self.reward_fn[agent_id].reset(positions)
        if self.config.get("mode") == "eval" and self.config.get("environment", {}).get("render_mode") == "human":
            print("Rendering reset (manual)")
            self.env.render()
        return obs_dict, info_dict

    def step(self, action_dict):
        self.step_count += 1
        self.episode_step_count += 1
        actions = np.zeros((len(self._agents), 2), dtype=np.float32)
        for i, agent_id in enumerate(self._agents):
            actions[i] = action_dict[agent_id]
            print(f"Agent {agent_id} action: speed={action_dict[agent_id][0]:.2f}, steering={action_dict[agent_id][1]:.2f}")
        obs, _, terminated, truncated, info = self.env.step(actions)
        print(f"Step obs: poses_x={obs['poses_x']}, poses_y={obs['poses_y']}, poses_theta={obs['poses_theta']}, collisions={obs['collisions']}, lap_counts={obs['lap_counts']}")
        print(f"Step info: {info}")
        rew_dict = {}
        obs_dict = {}
        for i, agent_id in enumerate(self._agents):
            agent_obs = np.concatenate([
                obs['scans'][i],
                np.array([
                    obs['poses_x'][i],
                    obs['poses_y'][i],
                    obs['poses_theta'][i]
                ])
            ])
            obs_dict[agent_id] = agent_obs.astype(np.float32)
            state = {
                "poses_x": obs['poses_x'],
                "poses_y": obs['poses_y'],
                "poses_theta": obs['poses_theta'],
                "linear_vels_x": obs['linear_vels_x'],
                "linear_vels_y": obs['linear_vels_y'],
                "ang_vels_z": obs['ang_vels_z'],
                "collisions": obs['collisions'],
                "lap_counts": obs['lap_counts'],
                "lap_times": obs['lap_times'],
                "actions": action_dict[agent_id],
                "prev_actions": self.env.prev_actions.get(agent_id, np.zeros(2)) if hasattr(self.env, 'prev_actions') else np.zeros(2),
            }
            reward, components = self.reward_fn[agent_id](state, agent_id)
            rew_dict[agent_id] = reward
            self.last_components[agent_id] = components
            self.episode_rewards[agent_id] += reward
            print(f"Agent {agent_id} reward: {reward:.2f}, components: {components}")
        terminated_dict = {agent_id: terminated for agent_id in self._agents}
        terminated_dict["__all__"] = terminated
        truncated_dict = {agent_id: truncated for agent_id in self._agents}
        truncated_dict["__all__"] = truncated
        info_dict = {agent_id: info for agent_id in self._agents}
        info_dict["__common__"] = {"episode_rewards": self.episode_rewards}
        print(f"Env state: terminated={terminated}, truncated={truncated}, episode_steps={self.episode_step_count}, lap_counts={obs['lap_counts']}, episode_rewards={self.episode_rewards}")
        if terminated or truncated:
            print(f"Episode terminated or truncated: terminated={terminated}, truncated={truncated}, episode_steps={self.episode_step_count}, lap_counts={obs['lap_counts']}, poses_x={obs['poses_x']}, poses_y={obs['poses_y']}, info={info}, total_rewards={self.episode_rewards}")
            self.episode_rewards = {agent_id: 0.0 for agent_id in self._agents}
        if self.config.get("mode") == "eval" and self.config.get("environment", {}).get("render_mode") == "human":
            print("Rendering step (manual)")
            self.env.render()
        return obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict

    def close(self):
        if hasattr(self.env, 'close'):
            print("Closing environment")
            self.env.close()

def custom_log_creator(log_dir):
    def logger_creator(config):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return UnifiedLogger(config, log_dir, loggers=None)
    return logger_creator

def setup_policies_and_config(config: Dict) -> tuple:
    print(f"setup_policies_and_config: Creating temp_env with config={config}")
    temp_env = MultiAgentF110(config)
    policies = {
        agent_id: (None, temp_env.observation_space[agent_id], temp_env.action_space[agent_id], {})
        for agent_id in temp_env.agents
    }
    algorithms = {}
    for agent_id in temp_env.agents:
        algo_config = config["algorithm"]["agents"][agent_id]
        algo_type = algo_config.pop("type")
        if algo_type == "ppo":
            ppo_config = ppo.PPOConfig().environment(env="f110_multi").env_runners(num_env_runners=2).resources(num_gpus=0).framework("torch")
            ppo_config.sgd_minibatch_size = 128
            ppo_config = ppo_config.training(**algo_config)
            algorithms[agent_id] = ppo_config
        elif algo_type == "sac":
            sac_config = sac.SACConfig().environment(env="f110_multi").env_runners(num_env_runners=2).resources(num_gpus=0).framework("torch")
            sac_config = sac_config.training(**algo_config)
            algorithms[agent_id] = sac_config
    return policies, algorithms

def setup_training(config: Dict):
    print(f"Config mode: {config['mode']}")  # Depuración
    print(f"Checkpoint path: {config.get('checkpoint_path', 'No checkpoint specified')}")
    ray.init(ignore_reinit_error=True)
    print("Ray initialized")
    
    
    
    # Pasar config completo al entorno registrado
    register_env("f110_multi", lambda env_config: MultiAgentF110(env_config))
    print("Environment registered")
    policies, algorithms = setup_policies_and_config(config)
    print(f"Policies: {policies}")
    log_dir = os.path.abspath(os.path.join("runs", f"multiagent_{int(time.time())}"))
    multiagent_config = {
        "policies": policies,
        "policy_mapping_fn": lambda agent_id, episode, worker=None, **kwargs: agent_id,
    }
    algo_config = algorithms["agent_0"].multi_agent(**multiagent_config)
    algo_config = algo_config.training(
        train_batch_size=config["algorithm"]["agents"]["agent_0"]["train_batch_size"]
    )
    algo_config = algo_config.env_runners(num_env_runners=2)
    algo_config = algo_config.resources(num_gpus=0)
    algo_config = algo_config.environment(env="f110_multi", env_config=config)
    algo_config = algo_config.reporting(keep_per_episode_custom_metrics=True)
    
   
    if config["mode"] == "eval":
        checkpoint_path = os.path.abspath(config["checkpoint_path"])
        algo = algo_config.build(logger_creator=custom_log_creator(log_dir))
        try:
            algo.restore(checkpoint_path)
            print(f"Restored checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to restore checkpoint: {str(e)}")
            exit(1)  # Exit if restoration fails
            raise
        
        
    else:
        algo = algo_config.build(logger_creator=custom_log_creator(log_dir))
        print("Algorithm built")

    total_timesteps = config["training"]["total_timesteps"]
    save_every = config["training"]["save_every"]
    save_best_metric = config["training"]["save_best_metric"]
    save_best_threshold = config["training"]["save_best_threshold"]
    best_metric = float("-inf")
    model_dir = os.path.abspath(os.path.join("models", f"multiagent_run_{int(time.time())}"))
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")
    try:
        for iteration in range(total_timesteps // save_every):
            print(f"Iteration {iteration}")
            try:
                result = algo.train()
                print(f"Raw result keys: {list(result.keys())}")
                per_agent_rewards = {}
                for agent_id in policies:
                    if f"policy_{agent_id}_reward_mean" in result:
                        print(f"Policy {agent_id} reward mean: {result[f'policy_{agent_id}_reward_mean']:.2f}")
                        per_agent_rewards[agent_id] = result[f'policy_{agent_id}_reward_mean']
                    elif f"env_runners/agent_episode_returns_mean/{agent_id}" in result:
                        print(f"Agent {agent_id} episode returns mean: {result[f'env_runners/agent_episode_returns_mean/{agent_id}']:.2f}")
                        per_agent_rewards[agent_id] = result[f'env_runners/agent_episode_returns_mean/{agent_id}']
                    else:
                        per_agent_rewards[agent_id] = 0.0
                episode_reward_mean = result.get("episode_reward_mean", 0.0)
                timesteps = result.get("timesteps_total", result.get("training_iteration", 0))
                print(f"Iteration: {iteration}, Timesteps/Iteration: {timesteps}, Per-Agent Rewards: {per_agent_rewards}, Avg Episode Reward: {episode_reward_mean:.2f}")
                if episode_reward_mean > best_metric and episode_reward_mean > save_best_threshold:
                    best_metric = episode_reward_mean
                    checkpoint_path = os.path.join(model_dir, "best")
                    checkpoint_uri = f"file://{checkpoint_path}"
                    print(f"Saving best checkpoint to: {checkpoint_uri}")
                    algo.save(checkpoint_uri)
                    print(f"New best model saved with {save_best_metric}: {best_metric:.2f}")
                if iteration % save_every == 0:
                    checkpoint_path = os.path.join(model_dir, f"checkpoint_{iteration}")
                    checkpoint_uri = f"file://{checkpoint_path}"
                    print(f"Saving checkpoint to: {checkpoint_uri}")
                    algo.save(checkpoint_uri)
                    print(f"Checkpoint saved at iteration {iteration}")
            except Exception as e:
                print(f"Error during iteration {iteration}: {str(e)}")
                raise
        final_checkpoint_path = os.path.join(model_dir, "final")
        final_checkpoint_uri = f"file://{final_checkpoint_path}"
        print(f"Saving final checkpoint to: {final_checkpoint_uri}")
        algo.save(final_checkpoint_uri)
        print(f"Training completed. Final checkpoints saved to {model_dir}")
        if best_metric > float("-inf"):
            print(f"Best model saved to {model_dir}/best with {save_best_metric}: {best_metric:.2f}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        algo.stop()
        print("Stopping algorithm")
        ray.shutdown()
        print("Ray shutdown")
""" 
def setup_training(config: Dict):
    print(f"Config mode: {config['mode']}")  # Depuración: confirmar modo
    print(f"Checkpoint path: {config.get('checkpoint_path', 'No checkpoint specified')}")  # Depuración: confirmar ruta del checkpoint
    ray.init(ignore_reinit_error=True)
    print("Ray initialized")
    # Pasar config completo al entorno registrado
    register_env("f110_multi", lambda env_config: MultiAgentF110(env_config))
    print("Environment registered")
    policies, algorithms = setup_policies_and_config(config)
    print(f"Policies: {policies}")
    log_dir = os.path.abspath(os.path.join("runs", f"multiagent_{int(time.time())}"))
    multiagent_config = {
        "policies": policies,
        "policy_mapping_fn": lambda agent_id, episode, worker=None, **kwargs: agent_id,
    }
    algo_config = algorithms["agent_0"].multi_agent(**multiagent_config)
    algo_config = algo_config.training(
        train_batch_size=config["algorithm"]["agents"]["agent_0"]["train_batch_size"]
    )
    algo_config = algo_config.env_runners(num_env_runners=2)
    algo_config = algo_config.resources(num_gpus=0)
    algo_config = algo_config.environment(env="f110_multi", env_config=config)
    algo_config = algo_config.reporting(keep_per_episode_custom_metrics=True)

    if config["mode"] == "eval":
        checkpoint_path = os.path.abspath(config["checkpoint_path"])
        print(f"Building algorithm for evaluation with checkpoint: {checkpoint_path}")
        algo = algo_config.build(logger_creator=custom_log_creator(log_dir))
        try:
            algo.restore(checkpoint_path)
            print(f"Restored checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to restore checkpoint: {str(e)}")
            raise
        # Ejecutar evaluación
        env = MultiAgentF110(config)
        try:
            obs, info = env.reset()
            done = {"__all__": False}
            total_rewards = {agent_id: 0.0 for agent_id in env.agents}
            step_count = 0
            while not done["__all__"]:
                # Calcular acciones para todos los agentes
                actions = algo.compute_actions(obs, policy_id=None)  # policy_id=None para multiagente
                obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated  # Usar terminated_dict para __all__
                for agent_id in env.agents:
                    total_rewards[agent_id] += rewards.get(agent_id, 0.0)
                step_count += 1
                print(f"Evaluation step {step_count}: Rewards={rewards}, Total Rewards={total_rewards}, Info={info}")
                # Depuración de renderizado
                if config.get("environment", {}).get("render_mode") == "human":
                    try:
                        env.render()
                        print("Render called successfully")
                    except Exception as e:
                        print(f"Render error: {str(e)}")
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
        finally:
            env.close()
            print(f"Evaluation completed. Total rewards: {total_rewards}")
            algo.stop()
            ray.shutdown()
            print("Ray shutdown")
    else:
        algo = algo_config.build(logger_creator=custom_log_creator(log_dir))
        print("Algorithm built")
        total_timesteps = config["training"]["total_timesteps"]
        save_every = config["training"]["save_every"]
        save_best_metric = config["training"]["save_best_metric"]
        save_best_threshold = config["training"]["save_best_threshold"]
        best_metric = float("-inf")
        model_dir = os.path.abspath(os.path.join("models", f"multiagent_run_{int(time.time())}"))
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        try:
            for iteration in range(total_timesteps // save_every):
                print(f"Iteration {iteration}")
                try:
                    result = algo.train()
                    print(f"Raw result keys: {list(result.keys())}")
                    per_agent_rewards = {}
                    for agent_id in policies:
                        if f"policy_{agent_id}_reward_mean" in result:
                            print(f"Policy {agent_id} reward mean: {result[f'policy_{agent_id}_reward_mean']:.2f}")
                            per_agent_rewards[agent_id] = result[f"policy_{agent_id}_reward_mean"]
                        elif f"env_runners/agent_episode_returns_mean/{agent_id}" in result:
                            print(f"Agent {agent_id} episode returns mean: {result[f'env_runners/agent_episode_returns_mean/{agent_id}']:.2f}")
                            per_agent_rewards[agent_id] = result[f"env_runners/agent_episode_returns_mean/{agent_id}"]
                        else:
                            per_agent_rewards[agent_id] = 0.0
                    episode_reward_mean = result.get("episode_reward_mean", 0.0)
                    timesteps = result.get("timesteps_total", result.get("training_iteration", 0))
                    print(f"Iteration: {iteration}, Timesteps/Iteration: {timesteps}, Per-Agent Rewards: {per_agent_rewards}, Avg Episode Reward: {episode_reward_mean:.2f}")
                    if episode_reward_mean > best_metric and episode_reward_mean > save_best_threshold:
                        best_metric = episode_reward_mean
                        checkpoint_path = os.path.join(model_dir, "best")
                        checkpoint_uri = f"file://{checkpoint_path}"
                        print(f"Saving best checkpoint to: {checkpoint_uri}")
                        algo.save(checkpoint_uri)
                        print(f"New best model saved with {save_best_metric}: {best_metric:.2f}")
                    if iteration % save_every == 0:
                        checkpoint_path = os.path.join(model_dir, f"checkpoint_{iteration}")
                        checkpoint_uri = f"file://{checkpoint_path}"
                        print(f"Saving checkpoint to: {checkpoint_uri}")
                        algo.save(checkpoint_uri)
                        print(f"Checkpoint saved at iteration {iteration}")
                except Exception as e:
                    print(f"Error during iteration {iteration}: {str(e)}")
                    raise
            final_checkpoint_path = os.path.join(model_dir, "final")
            final_checkpoint_uri = f"file://{final_checkpoint_path}"
            print(f"Saving final checkpoint to: {final_checkpoint_uri}")
            algo.save(final_checkpoint_uri)
            print(f"Training completed. Final checkpoints saved to {model_dir}")
            if best_metric > float("-inf"):
                print(f"Best model saved to {model_dir}/best with {save_best_metric}: {best_metric:.2f}")
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise
        finally:
            algo.stop()
            print("Stopping algorithm")
            ray.shutdown()
            print("Ray shutdown")
"""
            
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_test.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    setup_training(config)