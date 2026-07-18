#!/usr/bin/env python3
"""
DETAILED COLLISION TRACE ANALYSIS
==========================================
Deep dive into collision events to understand SAC model failure modes.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import f1tenth_gym
import time


def run_detailed_collision_analysis():
    print("🔍 DETAILED SAC COLLISION TRACE ANALYSIS")
    print("=" * 80)

    # Load trained SAC model
    model_path = 'models/sac_checkpoint_30000.zip'
    model = SAC.load(model_path)
    print(f"✅ Loaded SAC model from: {model_path}")

    # Create environment
    env = gym.make('f1tenth_gym:f1tenth-v0', map="Spielberg", num_agents=1)
    collision_episodes = []
    collision_threshold = 0

    # Run multiple episodes to catch collisions
    for episode in range(50):  # More episodes to catch collisions
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_reward = 0

        while not done and step_count < 200:
            # Extract observation (handle dict format)
            if isinstance(obs, dict):
                observation = obs['observation'] if 'observation' in obs else obs.get('scan', obs)
            else:
                observation = obs

            # Get action from model (deterministic for reproducibility)
            action, _ = model.predict(observation, deterministic=True)

            # Step environment
            obs_next, reward, done, truncated, info = env.step(action)

            # Track position and velocity
            pos_x = env.poses_x[0]
            pos_y = env.poses_y[0]
            pos_theta = env.poses_theta[0]
            vel_x = env.vel_x[0]
            collision = env.collisions[0]

            step_count += 1
            episode_reward += reward

            # Check for collision
            if collision > 0:
                print(f"\n🚨 COLLISION DETECTED - Episode {episode}, Step {step_count}")
                print(f"Final position: [{pos_x:.3f}, {pos_y:.3f}, {pos_theta:.3f}]")
                print(f"Final velocity: {vel_x:.3f} m/s")
                print(f"Final reward: {reward:.6f}")
                print(f"Action taken: {action}")
                print(f"Episode total reward: {episode_reward:.6f}")

                # Store collision episode data
                collision_data = {
                    'episode': episode,
                    'step': step_count,
                    'position': [pos_x, pos_y, pos_theta],
                    'velocity': vel_x,
                    'final_reward': reward,
                    'action': action.copy(),
                    'total_reward': episode_reward
                }
                collision_episodes.append(collision_data)
                break

            obs = obs_next

        if not done:
            print(f"✅ Episode {episode}: {step_count} steps, reward: {episode_reward:.3f}")

    print(f"\n📊 COLLISION SUMMARY:")
    print(f"Total episodes: 50")
    print(f"Collision episodes: {len(collision_episodes)}")
    print(f"Collision rate: {len(collision_episodes)/50*100:.1f}%")

    if collision_episodes:
        print(f"\n🔍 COLLISION DETAILS:")
        positions = [c['position'] for c in collision_episodes]
        velocities = [c['velocity'] for c in collision_episodes]
        rewards = [c['final_reward'] for c in collision_episodes]
        actions = [c['action'] for c in collision_episodes]

        print(f"Average collision position: {np.mean(positions, axis=0)}")
        print(f"Position std: {np.std(positions, axis=0)}")
        print(f"Average collision velocity: {np.mean(velocities):.3f} m/s")
        print(f"Velocity std: {np.std(velocities):.3f}")
        print(f"Average collision penalty: {np.mean(rewards):.6f}")
        print(f"Penalty std: {np.std(rewards):.6f}")
        print(f"Average collision action: {np.mean(actions, axis=0)}")
        print(f"Action std: {np.std(actions, axis=0)}")

        # Check if collision penalties match expected formula
        print(f"\n🧮 PENALTY FORMULA VERIFICATION:")
        for i, collision in enumerate(collision_episodes):
            vel = collision['velocity']
            expected_penalty = -0.05 * vel * vel
            actual_penalty = collision['final_reward']
            print(f"Collision {i+1}: v={vel:.2f}, expected={expected_penalty:.6f}, actual={actual_penalty:.6f}")

    env.close()


if __name__ == "__main__":
    run_detailed_collision_analysis()
