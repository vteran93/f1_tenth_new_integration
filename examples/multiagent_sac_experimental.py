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
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
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
        self.action_space = {agent: self._make_single_agent_action_space()
                             for agent in self.agents}

        # self.action_space = self._make_single_agent_action_space()
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
        return {agent: gym.spaces.Dict(single_spaces) for agent in self.agents}

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
        self.env.current_action = actions
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


class geminiReward(MultiAgentF110):
    """Custom reward function for Gemini."""

    def _get_rewards(self, newly_crashed):
        """
        Calculate individual rewards for each agent, combining track progress
        with incentives for survival and penalties for collisions.
        This version is improved by:
        - Using track progress as the primary reward (from Function 1).
        - Adding a survival reward for each step without a crash (inspired by Function 2).
        - Applying a more significant collision penalty (inspired by Function 2).
        - Scaling the progress reward to provide a stronger learning signal.
        - Correcting the lap completion logic.
        """

        # Initialize last_s tracking if not exists
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        rewards = []
        track_length = self.env.track.centerline.spline.s[-1]

        for i in range(self.env.num_agents):
            agent = self.agents[i]

            if agent in self._crashed_agents and agent not in newly_crashed:
                # Agent was already crashed - no reward
                reward = 0.0
            else:
                # Calculate track progress using centerline spline
                current_s, _ = self.env.track.raceline.spline.calc_arclength_inaccurate(
                    self.env.poses_x[i], self.env.poses_y[i]
                )

                # Calculate progress since last step
                prog = current_s - self._last_s[i]

                # Correctly handle lap completion (wrap-around)
                if prog < -0.5 * track_length:
                    # Crossed finish line going forward
                    prog += track_length
                elif prog > 0.5 * track_length:
                    # Crossed finish line going backward (unlikely but possible)
                    prog -= track_length

                # Apply rewards based on state
                if agent in newly_crashed:
                    # Strong penalty for collision, inspired by Function 2
                    reward = -5.0
                else:
                    # 1. Scaled progress reward (main incentive)
                    progress_reward = prog * 10.0

                    # 2. Small survival reward for each step not crashed (from Function 2)
                    survival_reward = 0.01

                    reward = progress_reward + survival_reward

                # Update last track position for this agent
                self._last_s[i] = current_s

            rewards.append(reward)

        return rewards


class geminiReward2(MultiAgentF110):
    """
    Función de recompensa rediseñada para guiar el aprendizaje de manera efectiva.

    Cambios Clave:
    1.  Se eliminó la gran penalización negativa por choque. En su lugar, el choque
        simplemente detiene la acumulación de recompensas.
    2.  Se agregó una recompensa por velocidad para incentivar el movimiento.
    3.  Se agregó una penalización por la distancia a la línea central para mantener
        al agente alejado de las paredes.
    4.  Se agregó una pequeña penalización por acciones de dirección bruscas para
        promover una conducción más suave.
    5.  Se volvió a usar 'centerline' para un cálculo de progreso más estable.
    """

    def __init__(self, env_config=None):
        super().__init__(env_config)
        # Cache para la línea central para evitar recalcularla
        self.centerline = self.env.track.centerline

    def _get_rewards(self, newly_crashed):
        """
        Calcula una recompensa densa para cada agente.
        """
        rewards = []

        for i in range(self.env.num_agents):
            agent = self.agents[i]

            current_pos = (self.env.poses_x[i], self.env.poses_y[i])

            if agent in self._crashed_agents and agent not in newly_crashed:
                # El agente ya había chocado, no hay recompensa.
                rewards.append(0.0)
                self._last_positions[i] = current_pos
                continue

            if agent in newly_crashed:
                # Penalización final solo en el momento del choque.
                # Es mucho más pequeña para no eclipsar otras recompensas.
                rewards.append(-1.0)
                self._last_positions[i] = current_pos
                continue

            # --- Recompensas para agentes activos ---

            # 1. Recompensa por Progreso (principal incentivo)
            # Usamos centerline para una referencia más estable
            current_s, _ = self.centerline.spline.calc_arclength_inaccurate(
                self.env.poses_x[i], self.env.poses_y[i]
            )
            # Asegúrate de que _last_s se inicialice
            if not hasattr(self, '_last_s'):
                self._last_s = [0.0] * self.env.num_agents
            prog = current_s - self._last_s[i]
            # Lógica de cruce de meta
            track_length = self.centerline.spline.s[-1]
            if prog < -0.5 * track_length:
                prog += track_length
            elif prog > 0.5 * track_length:
                prog -= track_length

            self._last_s[i] = current_s
            progress_reward = prog * 5.0  # Un poco menos de escala

            # 2. Recompensa por Velocidad
            last_pos = self._last_positions[i]
            distance = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
            speed = distance / self.env.timestep
            self._last_positions[i] = current_pos  # Actualizar para el siguiente paso
            speed_reward = speed * 0.1

            # 3. Penalización por distancia al centro de la pista
            # Esto ayuda a evitar las paredes
            _, nearest_dist = self.centerline.spline.find_closest_point(self.env.poses_x[i], self.env.poses_y[i])
            center_penalty = -nearest_dist * 0.2

            # 4. Penalización por acción de dirección
            # Fomenta una conducción más suave
            steering_action = self.env.current_action[i, 1]  # Asumiendo que el steering es el segundo elemento
            steering_penalty = -abs(steering_action) * 0.05

            # Recompensa total
            total_reward = progress_reward + speed_reward + center_penalty + steering_penalty
            rewards.append(total_reward)

        return rewards


class GPT4oMini(MultiAgentF110):
    """
    Política de recompensa multiagente para F110Env:
      - Máximo distancia recorrida
      - Bonus por cada vuelta completada
      - Penalización por accidentes (más fuerte si va a alta velocidad)
      - Penalización por alejarse de la raceline
      - Pequeño bonus por explorar nuevos tramos
    """

    def __init__(self,
                 env_config=None,
                 w_dist=1.0,
                 w_lap=100.0,
                 w_coll=-50.0,
                 w_coll_high=-100.0,
                 w_raceline=-1.0,
                 w_expl=0.1,
                 high_speed_thr=2.0):
        super().__init__(env_config)
        self.w_dist = w_dist
        self.w_lap = w_lap
        self.w_coll = w_coll
        self.w_coll_high = w_coll_high
        self.w_raceline = w_raceline
        self.w_expl = w_expl
        self.high_speed_thr = high_speed_thr

    def _get_rewards(self, newly_crashed):
        """
        states, next_states: list de dicts con keys 'distance','lap','speed','raceline_err','pos_id'
        infos: list de dicts con keys 'collision':bool, 'new_region':bool
        devuelve np.array shape=(n_agents,)
        """
        n = len(self.env.poses_x)
        rewards = np.zeros(n, dtype=np.float32)

        for i in range(n):
            s, ns, info = self.env.last_s[i], self.env.poses_x[i], self.env.observation_space[i]
            # 1) Distancia recorrida
            delta_dist = ns['distance'] - s['distance']
            r = self.w_dist * delta_dist

            # 2) Vueltas completadas
            lap_diff = ns['lap'] - s['lap']
            r += self.w_lap * lap_diff

            # 3) Colisión
            if newly_crashed[i]:
                if s['speed'] >= self.high_speed_thr:
                    r += self.w_coll_high
                else:
                    r += self.w_coll

            # 4) Proximidad a raceline (error lateral)
            r += self.w_raceline * abs(ns.get('raceline_err', 0.0))

            # 5) Exploración de pista
            if info.get('new_region', False):
                r += self.w_expl

            rewards[i] = r

        return rewards


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
        "reset_config": {"type": "rl_grid_static"},
        "render_mode": render_mode,
    }


active_class = GPT4oMini  # Default class for training/evaluation


def setup_policies_and_config():
    """Setup policies and SAC configuration. Returns (policies, config)."""
    register_env("f1tenth_multi", lambda config: active_class(config))

    # Create temporary environment for policy setup
    temp_env = active_class(get_env_config())
    policies = {agent: PolicySpec(None, temp_env.observation_space[agent], temp_env.action_space[agent], {})
                for agent in temp_env.agents}
    temp_env.close()

    # Configure SAC
    config = (SACConfig()
              .environment("f1tenth_multi", env_config=get_env_config())
              .framework("torch")
              .resources(num_gpus=1)  # Use 1 GPU if available)
              .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
              .env_runners(num_env_runners=10)
              .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
              .training(train_batch_size=4000,
                        replay_buffer_config={'type': 'MultiAgentPrioritizedReplayBuffer',
                                              'prioritized_replay_alpha': 0.6,
                                              'prioritized_replay_beta': 0.8,
                                              'prioritized_replay_eps': 1e-6,
                                              })
              )
    return policies, config


def custom_log_creator(custom_path: str, prefix: str):
    """
    Returns a function __logger_creator(config) that will
    write TensorBoard event files to `custom_path`.
    """
    def logger_creator(config):
        os.makedirs(custom_path, exist_ok=True)
        # The run_dir is now the custom_path itself, no extra timestamped subfolder
        return UnifiedLogger(config, custom_path, loggers=None)
    return logger_creator


def setup_ray_and_algo(config, run_name):
    """Initialize Ray and build algorithm—with TensorBoard logging."""
    ray.init(ignore_reinit_error=True)
    # Create a per-run log directory using the run_name
    log_dir = os.path.abspath(f"runs/{run_name}")
    os.makedirs(log_dir, exist_ok=True)

    # Build with a custom logger_creator to dump TB event files there
    return config.build(
        logger_creator=custom_log_creator(log_dir, "rllib_run")
    )


def setup_training(resume_from=None):
    """Setup and run training, with optional resume from a checkpoint."""
    print("Starting training...")

    model_dir = None
    run_name = None

    # Determine model directory for saving and restoring
    if resume_from:
        models_dir = "models"
        if resume_from is True:  # --resume without path, find latest
            print("Attempting to resume from the latest model...")
            if not os.path.exists(models_dir) or not os.listdir(models_dir):
                print("No models directory found or it's empty. Starting a new training run.")
            else:
                # Find all compatible run directories
                run_dirs = [d for d in os.listdir(models_dir) if d.startswith(
                    "multiagent_sac_") and os.path.isdir(os.path.join(models_dir, d))]
                if not run_dirs:
                    print("No compatible models found to resume from. Starting a new training run.")
                else:
                    # The timestamp is at the beginning of the name suffix, so lexicographical sort works
                    latest_model_dir_name = max(run_dirs)
                    model_dir = os.path.abspath(os.path.join(models_dir, latest_model_dir_name))
        else:  # --resume with a path
            model_dir = os.path.abspath(resume_from)
            if not os.path.exists(model_dir):
                print(f"Specified model path does not exist: {model_dir}. Starting a new training run.")
                model_dir = None  # Reset to start fresh

    # If resuming, get run_name from model_dir
    if model_dir:
        run_name = os.path.basename(model_dir)

    # If not resuming or resume path not found, create a new directory
    if not model_dir:
        env_config = get_env_config()
        reset_config = env_config["reset_config"]['type']
        env_class_name = active_class.__name__
        date_str = datetime.now().strftime("%Y%m%d%H%M%S")
        run_name = f"multiagent_sac_{date_str}_{reset_config}_{env_class_name}"

        model_dir = f"models/{run_name}"
        os.makedirs(model_dir, exist_ok=True)
        print(f"Starting new training run. Models will be saved to: {model_dir}")
    else:
        print(f"Resuming training. Models will be saved to existing directory: {model_dir}")

    # Setup policies and config
    policies, config = setup_policies_and_config()
    algo = setup_ray_and_algo(config, run_name)

    # Restore if we are resuming from an existing directory
    if resume_from and model_dir and os.path.exists(model_dir):
        try:
            # RLLib's restore can take the directory and finds the latest checkpoint
            algo.restore(model_dir)
            print(f"Successfully restored model from {model_dir}")
        except Exception as e:
            print(
                f"Could not restore model from {model_dir}. Training will start from scratch and overwrite. Error: {e}")

    # Training loop
    TOTAL_TIMESTEPS = 100_000

    while True:
        result = algo.train()
        timesteps_total = result['timesteps_total']

        print(f"Timesteps: {timesteps_total}")
        # Save to the determined model directory
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

    # Find all compatible run directories
    run_dirs = [d for d in os.listdir(models_dir) if d.startswith("multiagent_sac_")
                and os.path.isdir(os.path.join(models_dir, d))]
    if not run_dirs:
        print("No trained models found. Train a model first.")
        return

    # The timestamp is at the beginning of the name suffix, so lexicographical sort works
    latest_model = max(run_dirs)
    model_path = os.path.abspath(os.path.join(models_dir, latest_model))
    print(f"Using model: {latest_model}")

    # Setup algorithm
    policies, config = setup_policies_and_config()
    algo = setup_ray_and_algo(config, latest_model)
    algo.restore(model_path)

    # Run evaluation
    eval_env = active_class(get_env_config(render_mode="human"))

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
    parser.add_argument("--resume", nargs='?', const=True, default=None,
                        help="Resume training from the latest checkpoint, or a specific one if a path is provided.")
    args = parser.parse_args()

    if args.train:
        setup_training(resume_from=args.resume)
    else:
        setup_evaluation()

    ray.shutdown()
