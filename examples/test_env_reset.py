#!/usr/bin/env python3

"""
Test environment reset with explicit poses to understand where the reset system breaks.
"""

import numpy as np
import sys
import os
import gymnasium as gym

# Add the parent directory to sys.path to import f1tenth_gym
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_env_reset_with_explicit_poses():
    """Test environment reset with explicit poses."""
    print("=== TESTING ENVIRONMENT RESET WITH EXPLICIT POSES ===")
    # Create environment
    config = {
        "map": "Spielberg",
        "num_agents": 1,
        "timestep": 0.01,
        "integrator": "rk4",
        "control_input": ["speed", "steering_angle"],
        "model": "st",  # Use single-track model instead of multi-body
        "observation_config": {"type": "kinematic_state"},
        "reset_config": {"type": "rl_random_static"},
    }

    try:
        env = gym.make("f1tenth_gym:f1tenth-v0", config=config, render_mode=None)
        print("Environment created successfully")

        # Test multiple explicit pose resets
        test_poses = [
            np.array([[5.0, 10.0, 0.5]]),  # Test pose 1
            np.array([[15.0, 20.0, 1.0]]),  # Test pose 2
            np.array([[-5.0, -10.0, -0.5]]),  # Test pose 3
        ]

        for i, poses in enumerate(test_poses):
            print(f"\n--- Test {i+1}: Resetting with explicit poses {poses} ---")

            # Reset with explicit poses
            obs, info = env.reset(options={"poses": poses})

            # Check actual agent positions
            actual_poses = np.array([[env.poses_x[0], env.poses_y[0], env.poses_theta[0]]])
            print(f"Requested poses: {poses}")
            print(f"Actual poses:    {actual_poses}")

            # Check if poses match
            if np.allclose(poses, actual_poses, atol=1e-3):
                print("✓ Poses match!")
            else:
                print("⚠️  POSES DON'T MATCH!")

            # Also check simulator state
            sim_poses = env.sim.agent_poses
            print(f"Simulator poses: {sim_poses}")

        print("\n--- Testing automatic reset ---")
        # Test automatic reset (should use reset function)
        obs, info = env.reset()
        actual_poses = np.array([[env.poses_x[0], env.poses_y[0], env.poses_theta[0]]])
        print(f"Auto reset actual poses: {actual_poses}")

        # Reset again to see if it's different
        obs, info = env.reset()
        actual_poses2 = np.array([[env.poses_x[0], env.poses_y[0], env.poses_theta[0]]])
        print(f"Auto reset 2 actual poses: {actual_poses2}")

        if np.allclose(actual_poses, actual_poses2):
            print("⚠️  AUTO RESET PRODUCES IDENTICAL POSES - THIS IS THE BUG!")
        else:
            print("✓ Auto reset produces different poses")

        env.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_simulator_reset_directly():
    """Test simulator reset method directly."""
    print("\n=== TESTING SIMULATOR RESET DIRECTLY ===")

    from f1tenth_gym.envs.base_classes import Simulator
    from f1tenth_gym.envs.f110_env import F110Env
    # Create simulator directly
    params = F110Env.f1tenth_vehicle_params()
    sim = Simulator(
        params=params,
        num_agents=1,
        seed=42,
        num_beams=1080,
        action_type=None,  # Will use default
    )

    # Test poses
    test_poses = [
        np.array([[1.0, 2.0, 0.5]]),
        np.array([[5.0, 6.0, 1.0]]),
        np.array([[-1.0, -2.0, -0.5]]),
    ]

    for i, poses in enumerate(test_poses):
        print(f"\n--- Simulator Test {i+1}: {poses} ---")
        sim.reset(poses)

        # Check simulator state
        print(f"agent_poses: {sim.agent_poses}")
        print(f"agent state: {sim.agents[0].state}")

        # The agent state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # So positions should be in state[0:2] and yaw in state[4]
        actual_x = sim.agents[0].state[0]
        actual_y = sim.agents[0].state[1]
        actual_yaw = sim.agents[0].state[4]

        print(f"Agent actual pose: [{actual_x:.3f}, {actual_y:.3f}, {actual_yaw:.3f}]")

        if np.allclose([actual_x, actual_y, actual_yaw], poses[0], atol=1e-3):
            print("✓ Simulator reset worked correctly")
        else:
            print("⚠️  Simulator reset failed!")


if __name__ == "__main__":
    print("Testing F1TENTH environment reset with explicit poses...")

    test_simulator_reset_directly()
    test_env_reset_with_explicit_poses()

    print("\nEnvironment reset testing completed!")
