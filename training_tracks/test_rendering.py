#!/usr/bin/env python3
"""
Quick test script to verify that evaluation rendering works correctly.
"""

import os
import sys
import numpy as np
import time

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiagent_env import MultiAgentF110PPO
from rewards import get_reward_function


def test_eval_rendering():
    """Test that evaluation with rendering works smoothly."""
    print("🎬 Testing evaluation rendering...")
    
    # Setup environment with human rendering
    env_config = {
        "map": "oval_small",
        "num_agents": 2,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "original"},
        "reset_config": {"type": "cl_grid_static"},
        "render_mode": "human",
    }
    
    # Get reward function
    reward_function = get_reward_function("ppo", "default")
    
    # Create environment
    try:
        eval_env = MultiAgentF110PPO(env_config, reward_function=reward_function)
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False
    
    try:
        # Test reset
        obs, info = eval_env.reset(seed=42)
        print(f"✅ Environment reset successfully, got {len(obs)} agent observations")
        
        # Test a few steps with random actions
        for step in range(10):
            print(f"  Step {step + 1}/10")
            
            # Random actions for testing
            actions = {}
            for agent in obs.keys():
                # Simple forward motion with slight steering
                actions[agent] = [2.0, 0.1 * np.sin(step * 0.5)]
            
            # Step environment
            obs, rewards, terminated, truncated, info = eval_env.step(actions)
            
            # Render
            eval_env.render()
            
            # Small delay to see the animation
            time.sleep(0.05)
            
            # Check if done
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                print("  Episode finished early")
                break
        
        print("✅ Rendering test completed successfully")
        
        # Cleanup
        eval_env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during rendering test: {e}")
        eval_env.close()
        return False


if __name__ == "__main__":
    success = test_eval_rendering()
    if success:
        print("\n🎉 Rendering test passed! The evaluation should now work smoothly.")
    else:
        print("\n❌ Rendering test failed.")
        sys.exit(1)
