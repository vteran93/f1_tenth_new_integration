#!/usr/bin/env python3

"""
Debug script to analyze SAC model deterministic behavior and poor performance.

This script investigates:
1. Why SAC model evaluations behave deterministically
2. Why the model shows poor performance (48 steps, -6.00 reward)
3. Analysis of reset configuration impact
4. Comparison of deterministic vs non-deterministic evaluation
"""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
import f1tenth_gym


def analyze_reset_behavior():
    """Analyze how reset configuration affects starting positions."""
    print("="*80)
    print("ANALYZING RESET BEHAVIOR")
    print("="*80)

    # Test different reset configurations
    reset_configs = [
        {"type": "rl_random_static"},  # Used in training/evaluation
        {"type": "rl_random_random"},  # More randomized
        {"type": "rl_grid_static"},    # Grid-based static
    ]

    for config_name, reset_config in zip(["random_static", "random_random", "grid_static"], reset_configs):
        print(f"\n--- Testing {config_name} reset configuration ---")

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
                "reset_config": reset_config,
            },
        )

        # Test multiple resets to see if starting positions are deterministic
        print(f"Reset configuration: {reset_config}")
        start_positions = []
        for i in range(5):
            obs, info = env.reset()
            # Access poses directly from environment (not from observation)
            pose_x = env.poses_x[0]
            pose_y = env.poses_y[0]
            pose_theta = env.poses_theta[0]

            start_positions.append([pose_x, pose_y, pose_theta])
            print(f"  Reset {i+1}: x={pose_x:.4f}, y={pose_y:.4f}, theta={pose_theta:.4f}")

        start_positions = np.array(start_positions)

        # Check if positions are identical (deterministic)
        if len(start_positions) > 1:
            pos_std = np.std(start_positions, axis=0)
            is_deterministic = np.allclose(pos_std, 0, atol=1e-6)
            print(f"  Position standard deviation: {pos_std}")
            print(f"  Is deterministic: {is_deterministic}")

        env.close()


def analyze_sac_model_behavior():
    """Analyze SAC model prediction behavior."""
    print("\n" + "="*80)
    print("ANALYZING SAC MODEL BEHAVIOR")
    print("="*80)

    # Try to load the model
    model_path = "models/sac_checkpoint_30000.zip"

    try:
        print(f"Loading model from: {model_path}")
        model = SAC.load(model_path, device="cpu")
        print("✅ Model loaded successfully")

        # Create evaluation environment
        eval_env = gym.make(
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

        print(f"Action space: {eval_env.action_space}")
        print(f"Observation space: {eval_env.observation_space}")

        # Test deterministic vs non-deterministic predictions
        obs, info = eval_env.reset()

        print("\n--- Testing deterministic vs non-deterministic predictions ---")

        # Multiple deterministic predictions with same observation
        deterministic_actions = []
        for i in range(5):
            action, _states = model.predict(obs, deterministic=True)
            deterministic_actions.append(action.copy())
            print(f"Deterministic prediction {i+1}: {action}")

        deterministic_actions = np.array(deterministic_actions)
        det_std = np.std(deterministic_actions, axis=0)
        print(f"Deterministic action std: {det_std} (should be ~0)")

        # Multiple non-deterministic predictions with same observation
        print("\nNon-deterministic predictions:")
        stochastic_actions = []
        for i in range(5):
            action, _states = model.predict(obs, deterministic=False)
            stochastic_actions.append(action.copy())
            print(f"Stochastic prediction {i+1}: {action}")

        stochastic_actions = np.array(stochastic_actions)
        stoch_std = np.std(stochastic_actions, axis=0)
        print(f"Stochastic action std: {stoch_std} (should be >0)")

        eval_env.close()

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Creating a random policy for comparison...")

        # Test with random policy
        eval_env = gym.make(
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

        obs, info = eval_env.reset()
        print(f"Action space: {eval_env.action_space}")

        # Test random actions
        print("Random action samples:")
        for i in range(5):
            action = eval_env.action_space.sample()
            print(f"Random action {i+1}: {action}")

        eval_env.close()


def run_episode_analysis():
    """Run a detailed episode analysis to understand poor performance."""
    print("\n" + "="*80)
    print("EPISODE ANALYSIS")
    print("="*80)

    model_path = "models/sac_checkpoint_30000.zip"

    try:
        model = SAC.load(model_path, device="cpu")

        eval_env = gym.make(
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

        print("Running episode with detailed logging...")

        obs, info = eval_env.reset()
        done = False
        step_count = 0
        total_reward = 0
        collision_count = 0

        action_history = []
        reward_history = []

        while not done and step_count < 100:  # Limit to 100 steps for analysis
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            action_history.append(action.copy())

            # Take step
            obs, reward, done, trunc, info = eval_env.step(action)
            reward_history.append(reward)
            total_reward += reward
            step_count += 1

            # Check for collisions
            if info and 'collision' in info:
                if info['collision']:
                    collision_count += 1

            # Log every 10 steps or if done
            if step_count % 10 == 0 or done:
                print(f"Step {step_count}: action={action}, reward={reward:.3f}, done={done}")
                if info:
                    print(f"  Info: {info}")

        print(f"\n--- Episode Summary ---")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward per step: {total_reward/step_count:.4f}")
        print(f"Episode terminated: {done}")
        print(f"Collision count: {collision_count}")

        # Analyze action patterns
        if action_history:
            actions = np.array(action_history)
            print(f"\n--- Action Analysis ---")
            print(f"Action shape: {actions.shape}")
            print(f"Action mean: {np.mean(actions, axis=0)}")
            print(f"Action std: {np.std(actions, axis=0)}")
            print(f"Action min: {np.min(actions, axis=0)}")
            print(f"Action max: {np.max(actions, axis=0)}")

            # Check if actions are stuck
            action_changes = np.abs(np.diff(actions, axis=0))
            avg_action_change = np.mean(action_changes, axis=0)
            print(f"Average action change per step: {avg_action_change}")

        # Analyze reward patterns
        if reward_history:
            rewards = np.array(reward_history)
            print(f"\n--- Reward Analysis ---")
            print(f"Reward mean: {np.mean(rewards):.4f}")
            print(f"Reward std: {np.std(rewards):.4f}")
            print(f"Reward min: {np.min(rewards):.4f}")
            print(f"Reward max: {np.max(rewards):.4f}")

            # Show reward components breakdown
            print("First 10 step rewards:")
            for i, r in enumerate(rewards[:10]):
                print(f"  Step {i+1}: {r:.4f}")

        eval_env.close()

    except Exception as e:
        print(f"❌ Error in episode analysis: {e}")


def compare_with_random_policy():
    """Compare SAC model performance with random policy."""
    print("\n" + "="*80)
    print("COMPARING SAC vs RANDOM POLICY")
    print("="*80)

    model_path = "models/sac_checkpoint_30000.zip"

    # Test both policies
    for policy_name, use_model in [("Random Policy", False), ("SAC Model", True)]:
        print(f"\n--- Testing {policy_name} ---")

        try:
            if use_model:
                model = SAC.load(model_path, device="cpu")

            eval_env = gym.make(
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

            obs, info = eval_env.reset()
            done = False
            step_count = 0
            total_reward = 0

            while not done and step_count < 100:
                if use_model:
                    action, _states = model.predict(obs, deterministic=True)
                else:
                    action = eval_env.action_space.sample()

                obs, reward, done, trunc, info = eval_env.step(action)
                total_reward += reward
                step_count += 1

            print(f"  Steps: {step_count}")
            print(f"  Total reward: {total_reward:.3f}")
            print(f"  Avg reward/step: {total_reward/step_count:.4f}")
            print(f"  Episode done: {done}")

            eval_env.close()

        except Exception as e:
            print(f"  ❌ Error testing {policy_name}: {e}")


if __name__ == "__main__":
    print("🔍 SAC Model Deterministic Behavior Analysis")
    print("=" * 80)

    # Run all analyses
    analyze_reset_behavior()
    analyze_sac_model_behavior()
    run_episode_analysis()
    compare_with_random_policy()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    print("\n📋 KEY FINDINGS:")
    print("1. Deterministic behavior in evaluation is EXPECTED when using deterministic=True")
    print("2. Poor performance (48 steps, negative rewards) suggests:")
    print("   - Model may not be properly trained")
    print("   - Model might be colliding quickly")
    print("   - Reward function might be too harsh")
    print("   - Action space/observation mismatch")
    print("3. Check if model file exists and is valid")
    print("4. Compare training vs evaluation environment configurations")
