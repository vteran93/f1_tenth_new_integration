#!/usr/bin/env python3

"""
Compare reset behavior between different environment types to identify configuration differences.
"""

import gymnasium as gym
import numpy as np
import f1tenth_gym


def test_environment_reset_comparison():
    """Compare reset behavior between f1tenth-v0 and victor-multi-agent-v0."""
    print("="*80)
    print("COMPARING ENVIRONMENT RESET BEHAVIOR")
    print("="*80)

    # Common configuration
    base_config = {
        "map": "Spielberg",
        "num_agents": 1,
        "timestep": 0.01,
        "num_beams": 36,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "rl"},
        "reset_config": {"type": "rl_random_static"},
    }

    # Test different environment types
    env_types = [
        "f1tenth_gym:f1tenth-v0",
        "f1tenth_gym:victor-multi-agent-v0"
    ]

    for env_type in env_types:
        print(f"\n{'='*50}")
        print(f"TESTING: {env_type}")
        print(f"{'='*50}")

        try:
            # Create environment
            env = gym.make(env_type, config=base_config)
            print(f"✓ Environment {env_type} created successfully")

            # Test multiple resets
            poses_list = []
            for i in range(5):
                obs, info = env.reset()

                # Get agent poses - different methods for different envs
                if hasattr(env, 'poses_x'):
                    # Standard f1tenth environment
                    poses = np.array([[env.poses_x[0], env.poses_y[0], env.poses_theta[0]]])
                elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'poses_x'):
                    # Wrapped environment
                    poses = np.array(
                        [[env.unwrapped.poses_x[0], env.unwrapped.poses_y[0], env.unwrapped.poses_theta[0]]])
                elif hasattr(env, 'sim') and hasattr(env.sim, 'agent_poses'):
                    # Via simulator
                    poses = env.sim.agent_poses
                else:
                    print("⚠️  Cannot access agent poses")
                    poses = None

                if poses is not None:
                    poses_list.append(poses.flatten()[:3])  # First 3 elements (x, y, theta)
                    print(f"Reset {i+1}: {poses.flatten()[:3]}")

            env.close()

            # Check if all poses are identical (indicating deterministic behavior)
            if len(poses_list) > 1:
                poses_array = np.array(poses_list)
                all_identical = np.allclose(poses_array[0], poses_array[1:], atol=1e-6)

                if all_identical:
                    print(f"⚠️  ALL POSES IDENTICAL - DETERMINISTIC RESET BUG DETECTED!")
                    print(f"   All resets produce: {poses_array[0]}")
                else:
                    print(f"✓ Reset produces different poses - working correctly")
                    print(f"   Pose variation range:")
                    print(f"     X: [{poses_array[:,0].min():.3f}, {poses_array[:,0].max():.3f}]")
                    print(f"     Y: [{poses_array[:,1].min():.3f}, {poses_array[:,1].max():.3f}]")
                    print(f"     Theta: [{poses_array[:,2].min():.3f}, {poses_array[:,2].max():.3f}]")

        except Exception as e:
            print(f"❌ Error with {env_type}: {e}")
            import traceback
            traceback.print_exc()


def test_victor_env_internals():
    """Investigate the internal structure of victor-multi-agent environment."""
    print(f"\n{'='*50}")
    print("INVESTIGATING VICTOR-MULTI-AGENT ENVIRONMENT INTERNALS")
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

    try:
        env = gym.make("f1tenth_gym:victor-multi-agent-v0", config=config)
        print(f"Environment type: {type(env)}")
        print(f"Environment class: {env.__class__}")

        # Check if it's wrapped
        if hasattr(env, 'unwrapped'):
            print(f"Unwrapped type: {type(env.unwrapped)}")
            print(f"Unwrapped class: {env.unwrapped.__class__}")

        # Look for relevant attributes
        attrs = ['poses_x', 'poses_y', 'poses_theta', 'sim', 'reset_fn', '_reset_fn']
        for attr in attrs:
            if hasattr(env, attr):
                print(f"✓ Has attribute: {attr}")
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, attr):
                print(f"✓ Unwrapped has attribute: {attr}")
            else:
                print(f"⚠️  Missing attribute: {attr}")

        # Test reset behavior
        print("\n--- Testing reset behavior ---")
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Info: {info}")

        # Try to access reset function
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_reset_fn'):
            reset_fn = env.unwrapped._reset_fn
            print(f"Reset function: {reset_fn}")
            print(f"Reset function type: {type(reset_fn)}")

        env.close()

    except Exception as e:
        print(f"❌ Error investigating victor environment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_environment_reset_comparison()
    test_victor_env_internals()
