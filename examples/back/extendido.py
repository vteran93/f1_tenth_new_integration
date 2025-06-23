#!/usr/bin/env python3
"""
Script extendido para F1TENTH Gym: 
- Entrenamiento PPO multiagente
- Evaluación de políticas entrenadas
- Ejecución de PurePursuit para múltiples agentes
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import os
import time
import ray
import gymnasium as gym
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from f1tenth_gym.envs import F110Env
import imageio
from enum import Enum

# Workaround para compatibilidad con gymnasium
import gymnasium.envs.registration
class VectorizeMode(Enum):
    ASYNC = "async"
    SYNC = "sync"
gymnasium.envs.registration.VectorizeMode = VectorizeMode

# ======================== PurePursuitPlanner ========================
class PurePursuitPlanner:
    """Planificador PurePursuit para seguimiento de trayectorias."""
    def __init__(self, map_path, lookahead_distance, wheelbase, max_steer, max_speed, curvature_scale=1.0):
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.curvature_scale = curvature_scale
        self.waypoints = self.load_waypoints(map_path)
        self.current_index = 0

    def load_waypoints(self, map_path):
        """Carga waypoints desde archivo CSV."""
        raceline = pd.read_csv(map_path, sep=';', header=None, comment='#', engine='python')
        raceline.columns = ['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2']
        return raceline[['x_m', 'y_m']].values

    def compute_action(self, x, y, theta):
        """Calcula acción de control basada en posición actual."""
        # 1. Encontrar waypoint más cercano
        distances = np.linalg.norm(self.waypoints - np.array([x, y]), axis=1)
        closest_idx = np.argmin(distances)
        
        # 2. Buscar punto lookahead
        lookahead_idx = closest_idx
        while (lookahead_idx < len(self.waypoints) - 1 and 
               np.linalg.norm(self.waypoints[lookahead_idx] - np.array([x, y])) < self.lookahead_distance):
            lookahead_idx += 1
        
        lookahead_point = self.waypoints[lookahead_idx]
        
        # 3. Transformar a coordenadas locales
        dx = lookahead_point[0] - x
        dy = lookahead_point[1] - y
        local_x = np.cos(theta) * dx + np.sin(theta) * dy
        local_y = -np.sin(theta) * dx + np.cos(theta) * dy
        
        # 4. Calcular ángulo de giro
        gamma = 2.0 * local_y / (self.lookahead_distance**2)
        steer = np.arctan(gamma * self.wheelbase)
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        
        # 5. Calcular velocidad (puede ser constante o basada en curvatura)
        speed = self.max_speed
        
        return steer, speed

# ======================== Entorno Multiagente ========================
class SimpleMultiAgentF110(MultiAgentEnv):
    """Wrapper multiagente para F110Env con soporte para PurePursuit."""
    def __init__(self, env_config=None):
        config = env_config or {}
        self.env = F110Env(config=config)
        self.agents = [f"agent_{i}" for i in range(self.env.num_agents)]
        
        # Espacio de acción individual
        single_action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 10.0]), dtype=np.float32
        )
        
        # Espacio de observación individual
        original_obs_space = self.env.observation_space.spaces
        single_obs_spaces = {}
        for key, space in original_obs_space.items():
            if key == 'ego_idx':
                single_obs_spaces[key] = gym.spaces.Box(low=0, high=self.env.num_agents-1, shape=(), dtype=np.int32)
            elif key == 'scans':
                single_obs_spaces[key] = gym.spaces.Box(
                    low=space.low.min(), high=space.high.max(), shape=(space.shape[1],), dtype=space.dtype)
            else:
                single_obs_spaces[key] = gym.spaces.Box(
                    low=space.low.min(), high=space.high.max(), shape=(), dtype=space.dtype)
        
        self.action_space = single_action_space
        self.observation_space = gym.spaces.Dict(single_obs_spaces)
        self._last_positions = [(0.0, 0.0)] * self.env.num_agents

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_positions = [(self.env.poses_x[i], self.env.poses_y[i]) for i in range(self.env.num_agents)]
        
        # Preparar observaciones por agente
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            obs_dict[agent] = {
                'ego_idx': np.array(obs['ego_idx'], dtype=np.int32),
                'scans': np.array(obs['scans'][i], dtype=np.float32),
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

        # Recompensa basada en progreso
        rewards = []
        for i in range(self.env.num_agents):
            current_pos = (self.env.poses_x[i], self.env.poses_y[i])
            last_pos = self._last_positions[i]
            progress = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
            collision_penalty = 100.0 if self.env.collisions[i] else 0.0
            rewards.append(progress * 10.0 - collision_penalty + 0.1)
            self._last_positions[i] = current_pos

        # Preparar observaciones y estados terminales
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            obs_dict[agent] = {
                'ego_idx': np.array(obs['ego_idx'], dtype=np.int32),
                'scans': np.array(obs['scans'][i], dtype=np.float32),
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

# ======================== Funcionalidad PurePursuit ========================
def run_purepursuit(env_config, pp_config):
    """Ejecuta simulación con planificadores PurePursuit."""
    env = F110Env(config=env_config)
    num_agents = env_config['num_agents']
    
    # Inicializar planificadores
    planners = [
        PurePursuitPlanner(
            map_path=pp_config['map_path'],
            lookahead_distance=pp_config['lookahead_distance'],
            wheelbase=pp_config['wheelbase'],
            max_steer=pp_config['max_steer'],
            max_speed=pp_config['max_speed'],
            curvature_scale=pp_config.get('curvature_scale', 1.0)
        )
        for _ in range(num_agents)
    ]
    
    # Iniciar simulación
    obs, _ = env.reset()
    done = False
    step_count = 0
    max_steps = pp_config.get('max_steps', 10000)
    
    while not done and step_count < max_steps:
        actions_array = np.zeros((num_agents, 2), dtype=np.float32)
        
        for i in range(num_agents):
            x = obs['poses_x'][i]
            y = obs['poses_y'][i]
            theta = obs['poses_theta'][i]
            
            steer, speed = planners[i].compute_action(x, y, theta)
            actions_array[i] = [speed, steer]
        
        obs, _, done, _, _ = env.step(actions_array)
        env.render()
        step_count += 1
    
    env.close()
    print(f"Simulación PurePursuit completada en {step_count} pasos")

# ======================== Funciones PPO ========================
def train_ppo(env_config, train_config):
    """Entrena políticas PPO multiagente."""
    timestamp = str(int(time.time()))
    run_id = f"multiagent_ppo_run_{timestamp}"
    model_dir = f"models/{run_id}"
    os.makedirs(model_dir, exist_ok=True)

    def env_creator(config):
        return SimpleMultiAgentF110(config)

    register_env("f1tenth_multi", env_creator)

    # Configurar políticas
    temp_env = env_creator(env_config)
    policies = {
        agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        for agent in temp_env.agents
    }

    # Configurar PPO
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
    
    # Entrenamiento
    total_timesteps = train_config.get('total_timesteps', 20000)
    save_every = train_config.get('save_every', 2000)
    
    while True:
        result = algo.train()
        timesteps_total = result['timesteps_total']
        
        if timesteps_total % save_every == 0:
            print(f"Timesteps: {timesteps_total}")
            algo.save(model_dir)
        
        if timesteps_total >= total_timesteps:
            break
    
    # Guardar modelo final
    final_checkpoint = algo.save(model_dir)
    print(f"Modelo guardado en {final_checkpoint}")
    algo.stop()
    ray.shutdown()
    return model_dir

def evaluate_ppo(model_path, env_config):
    """Evalúa políticas PPO entrenadas."""
    def env_creator(config):
        return SimpleMultiAgentF110(config)

    register_env("f1tenth_multi", env_creator)
    
    # Configurar entorno temporal para inicialización
    temp_env = env_creator(env_config)
    policies = {
        agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        for agent in temp_env.agents
    }
    temp_env.close()

    # Configurar algoritmo
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
    algo.restore(model_path)
    
    # Crear entorno de evaluación
    eval_env_config = env_config.copy()
    eval_env_config["render_mode"] = "human"
    eval_env = env_creator(eval_env_config)
    
    # Ejecutar episodios de evaluación
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
        
        print(f"Episodio {episode + 1} completado en {step_count} pasos")
    
    eval_env.close()
    algo.stop()
    ray.shutdown()

# ======================== Función Principal ========================
def main():
    parser = argparse.ArgumentParser(description="F1TENTH Gym - Multiagente con PPO y PurePursuit")
    parser.add_argument('--config', type=str, required=True, help="Ruta al archivo YAML de configuración")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'purepursuit'], required=True,
                        help="Modo de operación: train, eval o purepursuit")
    args = parser.parse_args()

    # Cargar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    env_config = config['environment']
    
    if args.mode == 'train':
        train_config = config.get('ppo_training', {})
        train_ppo(env_config, train_config)
    
    elif args.mode == 'eval':
        models_dir = "models"
        run_dirs = [d for d in os.listdir(models_dir) if d.startswith("multiagent_ppo_run_")]
        
        if not run_dirs:
            print("No se encontraron modelos entrenados. Entrena primero con --mode train")
            exit(1)
        
        # Usar el modelo más reciente
        latest_model = max(run_dirs, key=lambda x: int(x.split("_")[-1]))
        model_path = os.path.abspath(os.path.join(models_dir, latest_model))
        print(f"Usando modelo: {latest_model}")
        
        evaluate_ppo(model_path, env_config)
    
    elif args.mode == 'purepursuit':
        pp_config = config['purepursuit']
        run_purepursuit(env_config, pp_config)

if __name__ == "__main__":
    main()
