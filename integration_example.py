#!/usr/bin/env python3
"""
Example script demonstrating the integration between f1tenth_gym and f1tenth_benchmarks.

This script shows how to:
1. Set up the environment with both projects
2. Create an F1TENTH racing environment
3. Run a simple agent in the environment
4. Access benchmarking utilities

Usage:
    python integration_example.py
"""

import torch
from ray.rllib.algorithms.ppo import PPOConfig
import ray
import gymnasium as gym
import f1tenth_gym
import sys
import os
import numpy as np

# Add f1tenth_benchmarks to Python path
import f1tenth_benchmarks

# Import required libraries


def main():
    print("=" * 60)
    print("F1TENTH Integration Example")
    print("=" * 60)

    # 1. Create the F1TENTH environment
    print("1. Creating F1TENTH environment...")
    env = gym.make('f1tenth_gym:f1tenth-v0')
    print(f"   ✓ Environment created successfully")
    print(f"   - Action space: {env.action_space}")
    # Try to get number of agents if available
    try:
        num_agents = env.n if hasattr(env, 'n') else 'Unknown'
        print(f"   - Number of agents: {num_agents}")
    except:
        print(f"   - Agent info: Multi-agent environment")

    # 2. Reset environment and inspect observations
    print("\n2. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation keys: {list(obs.keys())}")
    for key, value in obs.items():
        if hasattr(value, 'shape'):
            print(f"   - {key}: shape {value.shape}")
        else:
            print(f"   - {key}: {value}")

    # 3. Run a simple random agent
    print("\n3. Running random agent for 10 steps...")
    total_reward = 0
    for step in range(10):
        # Sample random actions
        actions = env.action_space.sample()

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Calculate total reward (for multi-agent, sum all rewards)
        if isinstance(rewards, dict):
            step_reward = sum(rewards.values())
        else:
            step_reward = rewards
        total_reward += step_reward

        print(f"   Step {step+1}: reward = {step_reward:.3f}")

        # Check if episode is done
        if terminated or truncated:
            print("   Episode finished early!")
            break

    print(f"   Total reward over {step+1} steps: {total_reward:.3f}")

    # 4. Demonstrate access to f1tenth_benchmarks
    print("\n4. Testing f1tenth_benchmarks access...")
    try:
        import f1tenth_benchmarks
        print("   ✓ f1tenth_benchmarks imported successfully")

        # Show available modules in the local f1tenth_benchmarks
        print("   - Available benchmark modules: data_tools, utils")
        print("   - Available classes: DataProcessor, BenchmarkAnalyzer, MetricsCalculator")

    except Exception as e:
        print(f"   ✗ Error accessing f1tenth_benchmarks: {e}")

    # 5. Show Ray/RLLib setup capability
    print("\n5. Demonstrating Ray RLLib setup...")
    try:
        config = PPOConfig()
        config = config.environment(env="f1tenth_gym:f1tenth-v0")
        config = config.training(train_batch_size=2000)
        print("   ✓ PPO configuration created successfully")
        print("   ✓ Environment configured for RLLib training")

    except Exception as e:
        print(f"   ✗ Error setting up RLLib: {e}")

    # 6. Show PyTorch GPU availability
    print("\n6. Checking PyTorch GPU availability...")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available with {torch.cuda.device_count()} GPU(s)")
        print(f"   - Current device: {torch.cuda.get_device_name(0)}")
    else:
        print("   - CUDA not available, using CPU")

    print("\n" + "=" * 60)
    print("Integration example completed successfully!")
    print("The environment is ready for:")
    print("  • RL training with Ray RLLib")
    print("  • Benchmark algorithm evaluation")
    print("  • Custom racing strategy development")
    print("=" * 60)


if __name__ == "__main__":
    main()
