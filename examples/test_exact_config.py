#!/usr/bin/env python3

"""
Test the exact same configuration as used in the SAC debug script to isolate the issue.
"""

import gymnasium as gym
import numpy as np
import f1tenth_gym


def test_exact_sac_config():
    """Test the exact configuration used in the SAC debug script."""
    print("="*80)
    print("TESTING EXACT SAC DEBUG CONFIGURATION")
    print("="*80)

    # This is exactly what's used in debug_sac_deterministic.py
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

    print("Environment created successfully")
    print(f"Environment type: {type(env)}")
    print(f"Unwrapped type: {type(env.unwrapped)}")

    # Test multiple resets like in the debug script
    poses_list = []
    for i in range(5):
        obs, info = env.reset()

        # Access poses exactly like in debug script
        actual_poses = np.array([[env.poses_x[0], env.poses_y[0], env.poses_theta[0]]])
        poses_list.append(actual_poses.flatten())
        print(f"Reset {i+1}: {actual_poses.flatten()}")

    env.close()

    # Check determinism
    poses_array = np.array(poses_list)
    all_identical = np.allclose(poses_array[0], poses_array[1:], atol=1e-6)

    if all_identical:
        print(f"⚠️  DETERMINISTIC BEHAVIOR DETECTED!")
        print(f"   All resets produce: {poses_array[0]}")
        return True
    else:
        print(f"✓ Reset produces different poses")
        return False


def test_with_different_seeds():
    """Test if seeding affects reset behavior."""
    print(f"\n{'='*50}")
    print("TESTING WITH DIFFERENT SEEDS")
    print(f"{'='*50}")

    config = {
        "map": "Spielberg",
        "num_agents": 1,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "rl"},
        "reset_config": {"type": "rl_random_static"},
    }

    for seed in [None, 42, 123, 999]:
        print(f"\n--- Testing with seed: {seed} ---")

        env = gym.make("f1tenth_gym:victor-multi-agent-v0", config=config)

        # Set seed if provided
        if seed is not None:
            env.reset(seed=seed)

        # Test multiple resets
        poses_list = []
        for i in range(3):
            obs, info = env.reset()
            actual_poses = np.array([[env.poses_x[0], env.poses_y[0], env.poses_theta[0]]])
            poses_list.append(actual_poses.flatten())
            print(f"  Reset {i+1}: {actual_poses.flatten()}")

        env.close()

        # Check if seed makes it deterministic
        poses_array = np.array(poses_list)
        all_identical = np.allclose(poses_array[0], poses_array[1:], atol=1e-6)

        if all_identical:
            print(f"  ⚠️  Seed {seed} causes deterministic behavior")
        else:
            print(f"  ✓ Seed {seed} allows varied poses")


def test_isolated_reset_function():
    """Test the reset function in isolation."""
    print(f"\n{'='*50}")
    print("TESTING RESET FUNCTION IN ISOLATION")
    print(f"{'='*50}")

    # Create environment and extract reset function
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

    # Access the reset function
    reset_fn = env.reset_fn
    print(f"Reset function: {reset_fn}")
    print(f"Reset function type: {type(reset_fn)}")

    # Test reset function directly
    print("\n--- Testing reset function directly ---")
    for i in range(5):
        poses = reset_fn(1)  # 1 agent
        print(f"Direct reset {i+1}: {poses}")

    env.close()


if __name__ == "__main__":
    is_deterministic = test_exact_sac_config()
    test_with_different_seeds()
    test_isolated_reset_function()

    if is_deterministic:
        print(f"\n{'='*80}")
        print("DETERMINISTIC BEHAVIOR CONFIRMED!")
        print("Need to investigate further...")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("RESET SYSTEM WORKING CORRECTLY")
        print("The issue may be elsewhere...")
        print(f"{'='*80}")
