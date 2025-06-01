#!/usr/bin/env python3
"""
Quick check of observation structure
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import f1tenth_gym


def check_obs_structure():
    print("🔍 CHECKING OBSERVATION STRUCTURE")
    print("=" * 50)

    # Create environment
    env = gym.make('f1tenth_gym:f1tenth-v0', map="Spielberg", num_agents=1)

    # Reset and get observation
    obs, info = env.reset()

    print(f"Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            print(f"  {key}: type={type(value)}, shape={np.array(value).shape if hasattr(value, 'shape') or isinstance(value, (list, tuple)) else 'N/A'}")
    else:
        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"First few values: {obs[:5]}")

    env.close()


if __name__ == "__main__":
    check_obs_structure()
