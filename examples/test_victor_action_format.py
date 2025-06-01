#!/usr/bin/env python3
"""
Test Victor Environment Action Format
"""

import gymnasium as gym
import numpy as np
import f1tenth_gym


def test_victor_action_format():
    print("🔍 TESTING VICTOR ENVIRONMENT ACTION FORMAT")
    print("=" * 60)

    # Test the exact same environment configuration as in training/evaluation
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

    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")

    # Reset environment
    obs, info = env.reset()

    print(f"Observation type: {type(obs)}")
    print(f"Observation length: {len(obs) if isinstance(obs, (list, tuple)) else 'N/A'}")
    if isinstance(obs, (list, tuple)) and len(obs) > 0:
        print(f"First agent obs shape: {obs[0].shape}")

    # Test different action formats
    print("\n🧪 TESTING ACTION FORMATS:")

    # Test 1: Single action (steering, speed)
    try:
        action1 = np.array([0.1, 5.0])  # steering, speed
        print(f"Test 1 - Single action {action1.shape}: {action1}")
        obs1, reward1, done1, trunc1, info1 = env.step(action1)
        print(f"  ✅ Success: reward={reward1}, done={done1}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Reset for next test
    obs, info = env.reset()

    # Test 2: Action wrapped in list
    try:
        action2 = [np.array([0.1, 5.0])]  # List of actions
        print(f"Test 2 - Action in list: {action2}")
        obs2, reward2, done2, trunc2, info2 = env.step(action2)
        print(f"  ✅ Success: reward={reward2}, done={done2}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Reset for next test
    obs, info = env.reset()

    # Test 3: 2D array format
    try:
        action3 = np.array([[0.1, 5.0]])  # 2D array format
        print(f"Test 3 - 2D array {action3.shape}: {action3}")
        obs3, reward3, done3, trunc3, info3 = env.step(action3)
        print(f"  ✅ Success: reward={reward3}, done={done3}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    env.close()


if __name__ == "__main__":
    test_victor_action_format()
