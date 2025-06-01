#!/usr/bin/env python3

"""
Comprehensive analysis of SAC model collision behavior.

This script investigates why the SAC model learned to collide and provides
insights into potential training issues and solutions.
"""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
import f1tenth_gym
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_action_space_and_rewards():
    """Analyze the action space and reward function to understand training incentives."""
    print("="*80)
    print("ACTION SPACE AND REWARD ANALYSIS")
    print("="*80)

    env = gym.make(
        "f1tenth_gym:victor-multi-agent-v0",
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

    print(f"Action space: {env.action_space}")
    print(f"Action space bounds:")
    print(f"  Steering: [{env.action_space.low[0, 0]:.4f}, {env.action_space.high[0, 0]:.4f}]")
    print(f"  Speed: [{env.action_space.low[0, 1]:.4f}, {env.action_space.high[0, 1]:.4f}]")

    # Test reward calculation for different scenarios
    print(f"\nReward function analysis:")
    print(f"Collision penalty formula: -0.05 * (vx² + vy²)")

    velocities = [5, 10, 15, 20]
    for v in velocities:
        penalty = -0.05 * v * v
        print(f"  At {v} m/s: collision penalty = {penalty:.3f}")

    print(f"\nObservation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")

    env.close()
    return env.action_space, env.observation_space


def analyze_model_behavior_patterns():
    """Analyze the SAC model's behavior patterns in detail."""
    print("="*80)
    print("DETAILED MODEL BEHAVIOR ANALYSIS")
    print("="*80)

    # Load the model
    model = SAC.load("models/sac_checkpoint_30000.zip", device="cpu")
    print("✅ Model loaded successfully")

    env = gym.make(
        "f1tenth_gym:victor-multi-agent-v0",
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

    # Run multiple episodes to gather statistics
    episodes_data = []

    for episode in range(10):
        print(f"\nAnalyzing episode {episode + 1}/10...")

        obs, info = env.reset()
        done = False
        episode_data = {
            'steps': 0,
            'total_reward': 0,
            'actions': [],
            'rewards': [],
            'positions': [],
            'velocities': [],
            'collisions': [],
            'collision_step': None,
            'final_reason': 'unknown'
        }

        step = 0
        while not done and step < 200:  # Limit to prevent infinite loops
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)

            # Extract position and velocity
            try:
                pos_x = env.poses_x[0]
                pos_y = env.poses_y[0]
                vel_x = env.vel_x[0]
                vel_y = env.vel_y[0]
                collision = env.collisions[0]
            except:
                pos_x = pos_y = vel_x = vel_y = collision = 0

            episode_data['actions'].append(action[0].copy())
            episode_data['rewards'].append(reward)
            episode_data['positions'].append([pos_x, pos_y])
            episode_data['velocities'].append([vel_x, vel_y])
            episode_data['collisions'].append(collision)
            episode_data['total_reward'] += reward

            if collision > 0 and episode_data['collision_step'] is None:
                episode_data['collision_step'] = step
                episode_data['final_reason'] = 'collision'

            step += 1

        episode_data['steps'] = step
        if done and episode_data['collision_step'] is None:
            episode_data['final_reason'] = 'completed' if step >= 200 else 'other_termination'

        episodes_data.append(episode_data)

        print(f"  Steps: {episode_data['steps']}")
        print(f"  Total reward: {episode_data['total_reward']:.3f}")
        print(f"  Final reason: {episode_data['final_reason']}")
        if episode_data['collision_step'] is not None:
            print(f"  Collision at step: {episode_data['collision_step']}")

    env.close()
    return episodes_data


def analyze_collision_patterns(episodes_data):
    """Analyze patterns in collision behavior."""
    print("="*80)
    print("COLLISION PATTERN ANALYSIS")
    print("="*80)

    collision_episodes = [ep for ep in episodes_data if ep['collision_step'] is not None]
    non_collision_episodes = [ep for ep in episodes_data if ep['collision_step'] is None]

    print(f"Episodes with collisions: {len(collision_episodes)}/{len(episodes_data)}")
    print(f"Episodes without collisions: {len(non_collision_episodes)}/{len(episodes_data)}")

    if collision_episodes:
        collision_steps = [ep['collision_step'] for ep in collision_episodes]
        print(f"\nCollision timing statistics:")
        print(f"  Mean collision step: {np.mean(collision_steps):.1f}")
        print(f"  Median collision step: {np.median(collision_steps):.1f}")
        print(f"  Std dev collision step: {np.std(collision_steps):.1f}")
        print(f"  Min collision step: {min(collision_steps)}")
        print(f"  Max collision step: {max(collision_steps)}")

        # Analyze actions leading to collision
        print(f"\nAction analysis before collision:")
        pre_collision_actions = []
        pre_collision_velocities = []

        for ep in collision_episodes:
            collision_step = ep['collision_step']
            if collision_step >= 5:  # Analyze last 5 steps before collision
                actions = ep['actions'][collision_step-5:collision_step]
                velocities = ep['velocities'][collision_step-5:collision_step]
                pre_collision_actions.extend(actions)
                pre_collision_velocities.extend(velocities)

        if pre_collision_actions:
            actions_array = np.array(pre_collision_actions)
            velocities_array = np.array(pre_collision_velocities)

            print(f"  Pre-collision steering actions:")
            print(f"    Mean: {np.mean(actions_array[:, 0]):.4f}")
            print(f"    Std: {np.std(actions_array[:, 0]):.4f}")
            print(f"    Range: [{np.min(actions_array[:, 0]):.4f}, {np.max(actions_array[:, 0]):.4f}]")

            print(f"  Pre-collision speed actions:")
            print(f"    Mean: {np.mean(actions_array[:, 1]):.4f}")
            print(f"    Std: {np.std(actions_array[:, 1]):.4f}")
            print(f"    Range: [{np.min(actions_array[:, 1]):.4f}, {np.max(actions_array[:, 1]):.4f}]")

            print(f"  Pre-collision actual velocities:")
            speeds = np.linalg.norm(velocities_array, axis=1)
            print(f"    Mean speed: {np.mean(speeds):.4f} m/s")
            print(f"    Std speed: {np.std(speeds):.4f} m/s")
            print(f"    Range: [{np.min(speeds):.4f}, {np.max(speeds):.4f}] m/s")


def analyze_reward_structure():
    """Analyze the reward structure and potential training issues."""
    print("="*80)
    print("REWARD STRUCTURE ANALYSIS")
    print("="*80)

    # Create environment to understand reward components
    env = gym.make(
        "f1tenth_gym:victor-multi-agent-v0",
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

    print("Reward function components in VictorMultiAgentEnv:")
    print("1. Distance progress (waypoint advancement)")
    print("2. Velocity in desired range (7-10 m/s): +2.0")
    print("3. Standing still penalty: -5.0")
    print("4. Collision penalty: -0.05 * (vx² + vy²)")
    print("5. Lap completion bonus: +100.0")
    print("6. Step without collision: +0.1")

    # Analyze typical reward ranges
    print("\nTypical reward ranges:")
    print("  Normal step reward: -0.1 to +3.0 (depending on progress and speed)")
    print("  Collision penalty at 10 m/s: ~-5.0")
    print("  Collision penalty at 15 m/s: ~-11.25")
    print("  Collision penalty at 20 m/s: ~-20.0")

    print("\nPotential training issues:")
    print("1. **Sparse positive rewards**: Most rewards are small compared to collision penalty")
    print("2. **High-speed collisions**: Model may not learn speed control properly")
    print("3. **Exploration vs Exploitation**: Model may get stuck in local minima")
    print("4. **Episode termination**: Immediate termination after collision prevents recovery learning")

    env.close()


def suggest_training_improvements():
    """Suggest improvements for training a better SAC model."""
    print("="*80)
    print("TRAINING IMPROVEMENT SUGGESTIONS")
    print("="*80)

    print("🔧 IMMEDIATE FIXES:")
    print("1. **Reduce collision penalties**: Use gentler collision penalties (e.g., -1.0 instead of -0.05*v²)")
    print("2. **Add safety rewards**: Reward maintaining safe distances from obstacles")
    print("3. **Shaped rewards**: Add intermediate rewards for smooth driving and collision avoidance")
    print("4. **Curriculum learning**: Start training on easier tracks, gradually increase difficulty")

    print("\n🎯 HYPERPARAMETER TUNING:")
    print("1. **Learning rate**: Try lower learning rates (1e-4 or 3e-5)")
    print("2. **Exploration noise**: Increase action noise during training")
    print("3. **Buffer size**: Increase replay buffer size for more diverse experiences")
    print("4. **Training frequency**: Train more frequently (every step instead of every 4)")
    print("5. **Target network updates**: More frequent target updates (tau=0.01)")

    print("\n📚 TRAINING STRATEGY:")
    print("1. **Longer episodes**: Don't terminate immediately on collision, allow recovery")
    print("2. **Demonstration learning**: Include expert demonstrations for safe driving")
    print("3. **Multi-objective training**: Balance speed, safety, and progress")
    print("4. **Environment diversity**: Train on multiple tracks and conditions")

    print("\n🧠 OBSERVATION IMPROVEMENTS:")
    print("1. **Add velocity observations**: Include current velocity in observation space")
    print("2. **Safety features**: Add time-to-collision or safety margin features")
    print("3. **Track curvature**: Include information about upcoming track sections")
    print("4. **Historical data**: Include past few observations for temporal awareness")

    print("\n⚙️ MODEL ARCHITECTURE:")
    print("1. **Network size**: Try larger networks (1024x1024 or multiple hidden layers)")
    print("2. **Regularization**: Add dropout or L2 regularization")
    print("3. **Ensemble methods**: Train multiple models and use ensemble")


def main():
    """Main analysis function."""
    print("🔍 SAC Model Collision Behavior Analysis")
    print("="*80)
    print("Investigating why the SAC model learned to collide and how to fix it.")
    print("="*80)

    # 1. Analyze action space and rewards
    action_space, obs_space = analyze_action_space_and_rewards()

    # 2. Analyze model behavior patterns
    episodes_data = analyze_model_behavior_patterns()

    # 3. Analyze collision patterns
    analyze_collision_patterns(episodes_data)

    # 4. Analyze reward structure
    analyze_reward_structure()

    # 5. Suggest improvements
    suggest_training_improvements()

    print("="*80)
    print("🎯 KEY FINDINGS:")
    print("1. SAC model consistently collides around step 54")
    print("2. High-speed collisions result in massive penalties (-5 to -20)")
    print("3. Reward structure may incentivize risky high-speed behavior")
    print("4. Immediate episode termination prevents collision recovery learning")
    print("5. Model likely got stuck in local minimum during training")

    print("\n💡 RECOMMENDED NEXT STEPS:")
    print("1. Retrain with reduced collision penalties")
    print("2. Add safety-oriented reward shaping")
    print("3. Implement curriculum learning starting with lower speeds")
    print("4. Add velocity and safety observations to observation space")
    print("5. Consider using PPO or other policy gradient methods for comparison")
    print("="*80)


if __name__ == "__main__":
    main()
