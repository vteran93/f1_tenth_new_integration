#!/usr/bin/env python3
"""
FINAL COLLISION ANALYSIS - Correct Victor Environment Usage
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import f1tenth_gym


def run_final_collision_analysis():
    print("🔍 FINAL SAC COLLISION ANALYSIS - VICTOR ENVIRONMENT")
    print("=" * 80)

    # Load trained SAC model
    model_path = 'models/sac_checkpoint_30000.zip'
    model = SAC.load(model_path)
    print(f"✅ Loaded SAC model from: {model_path}")

    # Create EXACT same environment as used in training/evaluation
    env = gym.make(
        'f1tenth_gym:victor-multi-agent-v0',
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "rl"},
            "reset_config": {"type": "rl_random_static"},
        },
    )

    collision_episodes = []
    successful_episodes = []

    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")

    # Run multiple episodes to reproduce collision behavior
    for episode in range(50):  # More episodes to get good statistics
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        step_rewards = []

        while not done and step_count < 200:
            # Get action from model (deterministic for reproducibility)
            action, _ = model.predict(obs, deterministic=True)

            # Convert to correct format for victor environment (2D array)
            action_2d = action.reshape(1, -1)  # Shape (1, 2)

            # Step environment
            obs_next, reward, done, truncated, info = env.step(action_2d)

            step_count += 1
            episode_reward += reward
            step_rewards.append(reward)

            # Check for collision by looking at large negative reward
            if reward < -4.0:  # Large negative reward indicates collision
                print(f"\n🚨 COLLISION DETECTED - Episode {episode}, Step {step_count}")
                print(f"  Collision reward: {reward:.6f}")
                print(f"  Action taken: {action}")
                print(f"  Episode total reward: {episode_reward:.6f}")
                print(f"  Last 5 step rewards: {step_rewards[-5:]}")

                # Store collision episode data
                collision_data = {
                    'episode': episode,
                    'step': step_count,
                    'final_reward': reward,
                    'action': action.copy(),
                    'total_reward': episode_reward,
                    'step_rewards': step_rewards.copy()
                }
                collision_episodes.append(collision_data)
                break

            obs = obs_next

        # If episode completed without collision
        if not done and step_count < 200:
            print(f"✅ Episode {episode}: {step_count} steps, reward: {episode_reward:.3f}")
            successful_episodes.append({
                'episode': episode,
                'steps': step_count,
                'total_reward': episode_reward
            })
        elif step_count >= 200:
            print(f"🔄 Episode {episode}: Max steps reached, reward: {episode_reward:.3f}")
            successful_episodes.append({
                'episode': episode,
                'steps': step_count,
                'total_reward': episode_reward
            })

    # Detailed analysis
    print(f"\n" + "="*80)
    print("📊 COMPREHENSIVE COLLISION ANALYSIS")
    print("="*80)

    total_episodes = len(collision_episodes) + len(successful_episodes)
    print(f"Total episodes: {total_episodes}")
    print(f"Collision episodes: {len(collision_episodes)} ({len(collision_episodes)/total_episodes*100:.1f}%)")
    print(f"Successful episodes: {len(successful_episodes)} ({len(successful_episodes)/total_episodes*100:.1f}%)")

    if collision_episodes:
        print(f"\n🔍 COLLISION DETAILS:")
        collision_steps = [c['step'] for c in collision_episodes]
        collision_rewards = [c['final_reward'] for c in collision_episodes]
        collision_actions = [c['action'] for c in collision_episodes]

        print(f"Average collision step: {np.mean(collision_steps):.1f} ± {np.std(collision_steps):.1f}")
        print(f"Collision step range: {min(collision_steps)} - {max(collision_steps)}")
        print(f"Average collision penalty: {np.mean(collision_rewards):.6f} ± {np.std(collision_rewards):.6f}")
        print(f"Average collision action: {np.mean(collision_actions, axis=0)} ± {np.std(collision_actions, axis=0)}")

        # Check if collision penalties match velocity formula
        print(f"\n🧮 VELOCITY ANALYSIS:")
        for i, collision in enumerate(collision_episodes[:5]):  # Show first 5
            penalty = collision['final_reward']
            # Reverse-engineer velocity from penalty: penalty = -0.05 * v²
            estimated_velocity = np.sqrt(-penalty / 0.05) if penalty < 0 else 0
            print(f"Collision {i+1}: penalty={penalty:.6f}, estimated velocity={estimated_velocity:.2f} m/s")

    if successful_episodes:
        print(f"\n✅ SUCCESSFUL EPISODE ANALYSIS:")
        success_steps = [s['steps'] for s in successful_episodes]
        success_rewards = [s['total_reward'] for s in successful_episodes]

        print(f"Average successful episode length: {np.mean(success_steps):.1f} ± {np.std(success_steps):.1f} steps")
        print(f"Average successful episode reward: {np.mean(success_rewards):.3f} ± {np.std(success_rewards):.3f}")
        print(
            f"Best performing episode: {max(success_rewards):.3f} reward in {success_steps[success_rewards.index(max(success_rewards))]} steps")

    # Performance comparison with original poor results
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"Original poor performance: 48 steps, -6.00 total reward, -0.1250 avg reward/step")
    if collision_episodes:
        avg_collision_steps = np.mean([c['step'] for c in collision_episodes])
        avg_collision_reward = np.mean([c['total_reward'] for c in collision_episodes])
        print(
            f"Current collision episodes: {avg_collision_steps:.1f} steps, {avg_collision_reward:.2f} total reward, {avg_collision_reward/avg_collision_steps:.4f} avg reward/step")

    if successful_episodes:
        avg_success_steps = np.mean([s['steps'] for s in successful_episodes])
        avg_success_reward = np.mean([s['total_reward'] for s in successful_episodes])
        print(
            f"Current successful episodes: {avg_success_steps:.1f} steps, {avg_success_reward:.2f} total reward, {avg_success_reward/avg_success_steps:.4f} avg reward/step")

    env.close()

    return {
        'collision_rate': len(collision_episodes) / total_episodes,
        'collision_episodes': collision_episodes,
        'successful_episodes': successful_episodes
    }


if __name__ == "__main__":
    results = run_final_collision_analysis()
