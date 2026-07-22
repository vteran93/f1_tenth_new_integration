#!/usr/bin/env python3

"""
Targeted analysis of SAC collision behavior to reproduce and understand the specific
collision scenario observed in our earlier investigation.
"""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
import f1tenth_gym


def test_specific_collision_scenario():
    """Test the specific scenario where we observed collisions."""
    print("="*80)
    print("REPRODUCING SPECIFIC COLLISION SCENARIO")
    print("="*80)

    model = SAC.load("models/sac_checkpoint_30000.zip", device="cpu")

    # Test multiple times with same configuration as our previous investigations
    collision_count = 0
    total_tests = 20

    for test_num in range(total_tests):
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

        obs, info = env.reset()
        done = False
        step_count = 0
        collision_occurred = False
        collision_step = -1
        total_reward = 0

        # Get initial position
        try:
            initial_pos = [env.poses_x[0], env.poses_y[0], env.poses_theta[0]]
        except:
            initial_pos = [0, 0, 0]

        while not done and step_count < 100:  # Limit to 100 steps
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Check for collision
            try:
                if env.collisions[0] > 0 and not collision_occurred:
                    collision_occurred = True
                    collision_step = step_count
                    collision_count += 1

                    # Get collision details
                    pos_x = env.poses_x[0]
                    pos_y = env.poses_y[0]
                    vel_x = env.vel_x[0]
                    vel_y = env.vel_y[0]
                    speed = np.sqrt(vel_x**2 + vel_y**2)

                    print(f"🔥 COLLISION DETECTED - Test {test_num + 1}")
                    print(f"  Step: {collision_step}")
                    print(f"  Initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
                    print(f"  Collision position: [{pos_x:.3f}, {pos_y:.3f}]")
                    print(f"  Velocity at collision: [{vel_x:.3f}, {vel_y:.3f}] (speed: {speed:.3f})")
                    print(f"  Action at collision: {action[0]}")
                    print(f"  Reward at collision: {reward:.3f}")
                    print(f"  Total reward: {total_reward:.3f}")
                    break
            except:
                pass

        if not collision_occurred:
            print(f"✅ No collision - Test {test_num + 1}: {step_count} steps, reward: {total_reward:.3f}")

        env.close()

    print(f"\n📊 COLLISION STATISTICS:")
    print(f"  Collisions: {collision_count}/{total_tests} ({collision_count/total_tests*100:.1f}%)")
    print(
        f"  No collisions: {total_tests - collision_count}/{total_tests} ({(total_tests - collision_count)/total_tests*100:.1f}%)")

    return collision_count


def test_deterministic_reset_positions():
    """Test if specific reset positions consistently lead to collisions."""
    print("="*80)
    print("TESTING DETERMINISTIC RESET POSITIONS")
    print("="*80)

    model = SAC.load("models/sac_checkpoint_30000.zip", device="cpu")

    # Test with grid-based static reset to get more predictable starting positions
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
            "reset_config": {"type": "rl_grid_static"},  # More deterministic
        },
    )

    collision_positions = []
    safe_positions = []

    for test_num in range(10):
        obs, info = env.reset()
        done = False
        step_count = 0
        collision_occurred = False

        # Get initial position
        try:
            initial_pos = [env.poses_x[0], env.poses_y[0], env.poses_theta[0]]
        except:
            initial_pos = [0, 0, 0]

        while not done and step_count < 80:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            step_count += 1

            # Check for collision
            try:
                if env.collisions[0] > 0:
                    collision_occurred = True
                    collision_positions.append(initial_pos.copy())
                    print(
                        f"🔥 Collision from position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}] at step {step_count}")
                    break
            except:
                pass

        if not collision_occurred:
            safe_positions.append(initial_pos.copy())
            print(
                f"✅ Safe from position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}] - {step_count} steps")

    env.close()

    print(f"\n📊 POSITION ANALYSIS:")
    print(f"  Collision-prone positions: {len(collision_positions)}")
    print(f"  Safe positions: {len(safe_positions)}")

    if collision_positions:
        collision_positions = np.array(collision_positions)
        print(
            f"  Collision positions mean: [{np.mean(collision_positions[:, 0]):.3f}, {np.mean(collision_positions[:, 1]):.3f}, {np.mean(collision_positions[:, 2]):.3f}]")
        print(
            f"  Collision positions std: [{np.std(collision_positions[:, 0]):.3f}, {np.std(collision_positions[:, 1]):.3f}, {np.std(collision_positions[:, 2]):.3f}]")

    if safe_positions:
        safe_positions = np.array(safe_positions)
        print(
            f"  Safe positions mean: [{np.mean(safe_positions[:, 0]):.3f}, {np.mean(safe_positions[:, 1]):.3f}, {np.mean(safe_positions[:, 2]):.3f}]")
        print(
            f"  Safe positions std: [{np.std(safe_positions[:, 0]):.3f}, {np.std(safe_positions[:, 1]):.3f}, {np.std(safe_positions[:, 2]):.3f}]")


def test_action_consistency():
    """Test if the model produces consistent actions for the same observation."""
    print("="*80)
    print("TESTING ACTION CONSISTENCY")
    print("="*80)

    model = SAC.load("models/sac_checkpoint_30000.zip", device="cpu")

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
            "reset_config": {"type": "rl_grid_static"},
        },
    )

    obs, info = env.reset()

    # Test deterministic action prediction
    print("Testing deterministic action prediction:")
    det_actions = []
    for i in range(5):
        action, _states = model.predict(obs, deterministic=True)
        det_actions.append(action[0].copy())
        print(f"  Prediction {i+1}: {action[0]}")

    det_actions = np.array(det_actions)
    det_std = np.std(det_actions, axis=0)
    print(f"  Deterministic action std: {det_std} (should be ~0)")

    # Test stochastic action prediction
    print("\nTesting stochastic action prediction:")
    stoch_actions = []
    for i in range(5):
        action, _states = model.predict(obs, deterministic=False)
        stoch_actions.append(action[0].copy())
        print(f"  Prediction {i+1}: {action[0]}")

    stoch_actions = np.array(stoch_actions)
    stoch_std = np.std(stoch_actions, axis=0)
    print(f"  Stochastic action std: {stoch_std} (should be >0)")

    env.close()


def analyze_model_weights():
    """Analyze the model's learned weights and biases."""
    print("="*80)
    print("MODEL WEIGHT ANALYSIS")
    print("="*80)

    model = SAC.load("models/sac_checkpoint_30000.zip", device="cpu")

    # Get the policy network
    policy = model.policy

    print("Policy network architecture:")
    print(f"  Actor network: {policy.actor}")
    print(f"  Critic networks: {policy.critic}")

    # Analyze actor network weights
    if hasattr(policy.actor, 'mu'):
        print("\nActor network output layer (mu) weights:")
        mu_weights = policy.actor.mu.weight.data.numpy()
        mu_bias = policy.actor.mu.bias.data.numpy()

        print(f"  Weight shape: {mu_weights.shape}")
        print(f"  Weight mean: {np.mean(mu_weights):.6f}")
        print(f"  Weight std: {np.std(mu_weights):.6f}")
        print(f"  Weight range: [{np.min(mu_weights):.6f}, {np.max(mu_weights):.6f}]")

        print(f"  Bias: {mu_bias}")
        print(f"  Steering bias: {mu_bias[0]:.6f}")
        print(f"  Speed bias: {mu_bias[1]:.6f}")

        # Check if biases indicate speed preference
        if mu_bias[1] > 10:
            print("  ⚠️  HIGH SPEED BIAS DETECTED - Model favors high speeds!")
        elif mu_bias[1] < 0:
            print("  ⚠️  NEGATIVE SPEED BIAS - Model favors reverse/low speeds!")
        else:
            print("  ✅ Speed bias seems reasonable")


def main():
    """Main analysis function."""
    print("🎯 TARGETED SAC COLLISION ANALYSIS")
    print("="*80)
    print("Focused investigation of the specific collision behavior observed earlier.")
    print("="*80)

    # Test 1: Try to reproduce the collision scenario
    collision_count = test_specific_collision_scenario()

    # Test 2: Check if specific positions lead to collisions
    test_deterministic_reset_positions()

    # Test 3: Test action consistency
    test_action_consistency()

    # Test 4: Analyze model weights
    analyze_model_weights()

    print("="*80)
    print("🎯 TARGETED ANALYSIS SUMMARY:")
    print(f"1. Collision reproduction rate: {collision_count}/20 tests")
    print("2. Position-dependent behavior analyzed")
    print("3. Action consistency verified")
    print("4. Model weights examined for biases")
    print("="*80)


if __name__ == "__main__":
    main()
