#!/usr/bin/env python3
"""
Test script to verify that the integration between f1tenth_gym and f1tenth_benchmarks works correctly.
"""

import sys
import os

# Use local f1tenth_benchmarks module
import f1tenth_benchmarks

# Test basic imports


def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    try:
        import f1tenth_gym
        print("✓ f1tenth_gym imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import f1tenth_gym: {e}")
        return False

    try:
        import gymnasium as gym
        print("✓ gymnasium imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import gymnasium: {e}")
        return False

    try:
        import ray
        print("✓ ray imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ray: {e}")
        return False

    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False

    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False

    try:
        import f1tenth_benchmarks
        print("✓ f1tenth_benchmarks imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import f1tenth_benchmarks: {e}")
        return False

    return True


def test_environment_creation():
    """Test F1TENTH environment creation."""
    print("\nTesting environment creation...")

    try:
        import gymnasium as gym
        env = gym.make('f1tenth_gym:f1tenth-v0')
        print("✓ F1TENTH environment created successfully")

        # Test basic environment properties
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")

        # Test reset
        obs, info = env.reset()
        if isinstance(obs, dict):
            print(f"  - Reset successful, observation is dict with keys: {list(obs.keys())}")
        else:
            print(f"  - Reset successful, observation shape: {obs.shape}")

        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False


def test_ray_functionality():
    """Test Ray functionality."""
    print("\nTesting Ray functionality...")

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        print("✓ Ray RLLib imported successfully")

        # Test Ray initialization (without actually starting it to avoid conflicts)
        config = PPOConfig()
        print("✓ PPO config created successfully")

        return True
    except Exception as e:
        print(f"✗ Failed to test Ray functionality: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("F1TENTH Integration Test")
    print("=" * 60)

    # Run all tests
    tests_passed = 0
    total_tests = 3

    if test_imports():
        tests_passed += 1

    if test_environment_creation():
        tests_passed += 1

    if test_ray_functionality():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Integration is ready for development.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
