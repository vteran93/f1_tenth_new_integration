from typing import List, Dict, Any
import os
import numpy as np
import wandb
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# if using wandb (recommended):
# from wandb.integration.sb3 import WandbCallback

# Multi-agent configuration
NUM_AGENTS = 2  # Change this to the desired number of agents
# toggle this to train or evaluate
train = False


class MultiAgentCallback(BaseCallback):
    """Custom callback for multi-agent training logging"""

    def __init__(self, agent_id: int, verbose=0):
        super(MultiAgentCallback, self).__init__(verbose)
        self.agent_id = agent_id

    def _on_step(self) -> bool:
        # Log agent-specific metrics
        if self.n_calls % 1000 == 0:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                self.model.logger.record(f"agent_{self.agent_id}/timesteps", self.n_calls)
        return True


def create_multi_agent_env(num_agents: int, render_mode=None):
    """Create multi-agent F1Tenth environment"""
    config = {
        "map": "Spielberg",
        "num_agents": num_agents,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "rl"},
        "reset_config": {"type": "rl_random_static"},
    }

    if render_mode:
        config["render_mode"] = render_mode

    return gym.make("f1tenth_gym:f1tenth-v0", config=config)


def extract_agent_observation(obs, agent_id: int, num_agents: int):
    """Extract observation for a specific agent from multi-agent observation"""
    # Assuming observations are concatenated or in a specific format
    # This may need to be adjusted based on the actual observation structure
    if isinstance(obs, dict):
        return obs.get(f"agent_{agent_id}", obs)
    elif isinstance(obs, np.ndarray):
        # If observations are concatenated, split them
        obs_dim = len(obs) // num_agents
        start_idx = agent_id * obs_dim
        end_idx = (agent_id + 1) * obs_dim
        return obs[start_idx:end_idx]
    else:
        return obs  # Single agent case


def combine_actions(actions: List[np.ndarray]) -> np.ndarray:
    """Combine individual agent actions into multi-agent action"""
    return np.concatenate(actions)


if train:
    run = wandb.init(
        project="f1tenth_gym_sac_multiagent",
        sync_tensorboard=True,
        save_code=True,
    )

    # Create multi-agent environment
    env = create_multi_agent_env(NUM_AGENTS)

    # Create individual SAC models for each agent
    agents = []
    for agent_id in range(NUM_AGENTS):
        # Create a dummy single-agent environment to get action/observation spaces
        single_env = create_multi_agent_env(1)

        # Add action noise for better exploration
        n_actions = single_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        # Create SAC model for this agent
        agent_model = SAC(
            "MlpPolicy",
            single_env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}/agent_{agent_id}",
            device="cuda",
            seed=42 + agent_id,  # Different seed per agent
            action_noise=action_noise,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            policy_kwargs=dict(net_arch=[256, 256])
        )

        agents.append(agent_model)
        single_env.close()

    # Multi-agent training loop
    print(f"Starting multi-agent training with {NUM_AGENTS} agents...")

    # Training parameters
    total_timesteps = 1_000_000
    timesteps_per_agent = total_timesteps // NUM_AGENTS

    # Train each agent independently using experience from multi-agent environment
    obs, info = env.reset()

    for timestep in range(total_timesteps):
        # Get actions from all agents
        actions = []
        for agent_id in range(NUM_AGENTS):
            agent_obs = extract_agent_observation(obs, agent_id, NUM_AGENTS)
            action, _ = agents[agent_id].predict(agent_obs, deterministic=False)
            actions.append(action)

        # Combine actions and step environment
        combined_action = combine_actions(actions)
        next_obs, rewards, done, trunc, info = env.step(combined_action)

        # Train each agent with their individual experience
        for agent_id in range(NUM_AGENTS):
            agent_obs = extract_agent_observation(obs, agent_id, NUM_AGENTS)
            agent_next_obs = extract_agent_observation(next_obs, agent_id, NUM_AGENTS)
            agent_reward = rewards[agent_id] if isinstance(rewards, (list, np.ndarray)) else rewards

            # Store experience in agent's replay buffer
            agents[agent_id].replay_buffer.add(
                agent_obs, agent_next_obs, actions[agent_id], agent_reward, done, [info]
            )

            # Train the agent if enough experience collected
            if timestep > agents[agent_id].learning_starts and timestep % agents[agent_id].train_freq == 0:
                agents[agent_id]._train_step()

        obs = next_obs

        # Reset environment if episode ended
        if done or trunc:
            obs, info = env.reset()

        # Log progress
        if timestep % 10000 == 0:
            print(f"Timestep: {timestep}/{total_timesteps}")

    # Save all trained agents
    os.makedirs(f"models/{run.id}", exist_ok=True)
    for agent_id, agent in enumerate(agents):
        agent.save(f"models/{run.id}/agent_{agent_id}_model.zip")

    print("Multi-agent training completed!")
    run.finish()

else:
    # Multi-agent evaluation
    print(f"Starting multi-agent evaluation with {NUM_AGENTS} agents...")

    # Load trained agents
    agents = []
    for agent_id in range(NUM_AGENTS):
        model_path = f"models/3wlusg06/agent_{agent_id}_model.zip"

        # Check if agent-specific model exists, otherwise use generic model
        if not os.path.exists(model_path):
            model_path = "models/3wlusg06/model.zip"  # Fallback to single agent model
            print(f"Agent {agent_id} model not found, using fallback: {model_path}")

        if os.path.exists(model_path):
            agent_model = SAC.load(model_path, print_system_info=(agent_id == 0), device="cpu")
            agents.append(agent_model)
            print(f"Loaded agent {agent_id} from {model_path}")
        else:
            print(f"Warning: No model found for agent {agent_id}")

    if not agents:
        print("No agents loaded. Please check model paths.")
        exit(1)

    # If we have fewer models than agents, duplicate the last model
    while len(agents) < NUM_AGENTS:
        agents.append(agents[-1])
        print(f"Duplicating agent model for agent {len(agents)-1}")

    # Create evaluation environment
    eval_env = create_multi_agent_env(NUM_AGENTS, render_mode="human")

    # Evaluation loop
    obs, info = eval_env.reset()
    done = False
    step_count = 0
    episode_rewards = [0.0] * NUM_AGENTS

    print("Starting evaluation episode...")

    while not done:
        # Get actions from all agents
        actions = []
        for agent_id in range(NUM_AGENTS):
            agent_obs = extract_agent_observation(obs, agent_id, NUM_AGENTS)
            action, _states = agents[agent_id].predict(agent_obs, deterministic=True)
            actions.append(action)

        # Combine actions and step environment
        combined_action = combine_actions(actions)
        obs, rewards, done, trunc, info = eval_env.step(combined_action)

        # Accumulate rewards for each agent
        if isinstance(rewards, (list, np.ndarray)):
            for agent_id in range(NUM_AGENTS):
                episode_rewards[agent_id] += rewards[agent_id]
        else:
            # Single reward for all agents
            for agent_id in range(NUM_AGENTS):
                episode_rewards[agent_id] += rewards / NUM_AGENTS

        eval_env.render()
        step_count += 1

        # Check for truncation as well
        if trunc:
            done = True

        # Optional: limit episode length for demonstration
        if step_count > 5000:  # Adjust as needed
            print("Episode truncated after 5000 steps")
            break

    print(f"Episode completed after {step_count} steps")
    print("Final episode rewards per agent:")
    for agent_id, reward in enumerate(episode_rewards):
        print(f"  Agent {agent_id}: {reward:.2f}")

    eval_env.close()
    print("Multi-agent evaluation completed!")
