#!/usr/bin/env python3
"""
FIXED COLLISION ANALYSIS - Using correct environment
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import f1tenth_gym


def run_collision_analysis_victor_env():
    print("🔍 COLLISION ANALYSIS WITH VICTOR ENVIRONMENT")
    print("=" * 80)

    # Load trained SAC model
    model_path = 'models/sac_checkpoint_30000.zip'
    model = SAC.load(model_path)
    print(f"✅ Loaded SAC model from: {model_path}")

    # Create the SAME environment used in training (victor-multi-agent)
    env = gym.make('f1tenth_gym:victor-multi-agent-v0', map="Spielberg", num_agents=1)

    collision_episodes = []

    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")

    # Run multiple episodes to catch collisions
    for episode in range(20):  # Start with fewer episodes for testing
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_reward = 0

        print(f"\nEpisode {episode}: obs length = {len(obs)}, type = {type(obs)}")
        if len(obs) > 0:
            print(f"  First agent obs shape = {obs[0].shape}")

        while not done and step_count < 200:
            # Get action from model using first agent's observation (single agent scenario)
            agent_obs = obs[0]  # Use first agent's observation
            action, _ = model.predict(agent_obs, deterministic=True)

            # Step environment
            obs_next, reward, done, truncated, info = env.step([action])  # Wrap action for multi-agent env

            step_count += 1
            episode_reward += reward[0] if isinstance(reward, (list, tuple)) else reward

            current_reward = reward[0] if isinstance(reward, (list, tuple)) else reward

            # Check for collision by looking at reward spike
            if current_reward < -4.0:  # Large negative reward indicates collision
                print(f"🚨 COLLISION DETECTED - Episode {episode}, Step {step_count}")
                print(f"Collision reward: {current_reward:.6f}")
                print(f"Action taken: {action}")
                print(f"Episode total reward: {episode_reward:.6f}")

                # Store collision episode data
                collision_data = {
                    'episode': episode,
                    'step': step_count,
                    'final_reward': current_reward,
                    'action': action.copy(),
                    'total_reward': episode_reward
                }
                collision_episodes.append(collision_data)
                break

            obs = obs_next

        if not done:
            print(f"✅ Episode {episode}: {step_count} steps, reward: {episode_reward:.3f}")
        elif step_count >= 200:
            print(f"🔄 Episode {episode}: Max steps reached, reward: {episode_reward:.3f}")

    print(f"\n📊 COLLISION SUMMARY:")
    print(f"Total episodes: 20")
    print(f"Collision episodes: {len(collision_episodes)}")
    print(f"Collision rate: {len(collision_episodes)/20*100:.1f}%")

    if collision_episodes:
        print(f"\n🔍 COLLISION DETAILS:")
        rewards = [c['final_reward'] for c in collision_episodes]
        actions = [c['action'] for c in collision_episodes]

        print(f"Average collision penalty: {np.mean(rewards):.6f}")
        print(f"Penalty std: {np.std(rewards):.6f}")
        print(f"Average collision action: {np.mean(actions, axis=0)}")
        print(f"Action std: {np.std(actions, axis=0)}")

        for i, collision in enumerate(collision_episodes):
            print(
                f"Collision {i+1}: step={collision['step']}, reward={collision['final_reward']:.6f}, action={collision['action']}")

    env.close()


if __name__ == "__main__":
    run_collision_analysis_victor_env()
