#!/usr/bin/env python3

"""
Investigate the massive reward spike and early termination in SAC episodes.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import f1tenth_gym


def analyze_reward_spike():
    """Analyze what causes the massive -5.031 reward spike."""
    print("="*80)
    print("ANALYZING REWARD SPIKE AND EARLY TERMINATION")
    print("="*80)

    # Load SAC model
    model = SAC.load("models/sac_checkpoint_30000.zip", device="cpu")

    # Create environment
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

    print("Running episode with detailed state monitoring...")

    obs, info = env.reset()
    done = False
    step = 0

    while not done and step < 100:
        step += 1

        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Store pre-step state
        pre_x = env.poses_x[0]
        pre_y = env.poses_y[0]
        pre_theta = env.poses_theta[0]
        pre_vel_x = getattr(env, 'vel_x', [0])[0] if hasattr(env, 'vel_x') else 0
        pre_vel_y = getattr(env, 'vel_y', [0])[0] if hasattr(env, 'vel_y') else 0

        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

        # Store post-step state
        post_x = env.poses_x[0]
        post_y = env.poses_y[0]
        post_theta = env.poses_theta[0]

        # Calculate movement
        dx = post_x - pre_x
        dy = post_y - pre_y
        distance_moved = np.sqrt(dx**2 + dy**2)

        # Check for significant reward changes
        if reward < -1.0 or step >= 50:  # Log critical steps
            print(f"\nStep {step}: *** CRITICAL ***")
            print(f"  Action: {action.flatten()}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Done: {done}")
            print(f"  Pre-position:  [{pre_x:.4f}, {pre_y:.4f}, {pre_theta:.4f}]")
            print(f"  Post-position: [{post_x:.4f}, {post_y:.4f}, {post_theta:.4f}]")
            print(f"  Movement: dx={dx:.4f}, dy={dy:.4f}, dist={distance_moved:.4f}")
            print(f"  Info: {info}")

            # Check collision status
            if hasattr(env, 'collisions'):
                print(f"  Collisions: {env.collisions}")
            if hasattr(env, 'sim') and hasattr(env.sim, 'collisions'):
                print(f"  Sim collisions: {env.sim.collisions}")

            # Check if agent is off track
            if hasattr(env, 'track'):
                # Try to determine if position is on track
                print(f"  Track info available: {hasattr(env.track, 'raceline')}")

        elif step % 10 == 0:  # Log every 10 steps
            print(
                f"Step {step}: reward={reward:.4f}, pos=[{post_x:.2f}, {post_y:.2f}], dist_moved={distance_moved:.4f}")

    print(f"\nEpisode ended at step {step}")
    print(f"Final done status: {done}")

    env.close()


def compare_reward_functions():
    """Compare reward calculation in different scenarios."""
    print(f"\n{'='*50}")
    print("COMPARING REWARD CALCULATION")
    print(f"{'='*50}")

    # Test multiple episodes to see reward patterns
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
            "reset_config": {"type": "rl_random_static"},
        },
    )

    for episode in range(3):
        print(f"\n--- Episode {episode+1} ---")
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0

        while not done and step < 60:  # Run until termination or 60 steps
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            total_reward += reward

            if done or step % 15 == 0:
                print(f"  Step {step}: reward={reward:.4f}, total={total_reward:.4f}, done={done}")

        print(f"Episode {episode+1} ended: {step} steps, {total_reward:.4f} total reward")

    env.close()


def test_static_actions():
    """Test how the environment responds to static actions."""
    print(f"\n{'='*50}")
    print("TESTING STATIC ACTIONS")
    print(f"{'='*50}")

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

    # Test different action combinations
    test_actions = [
        [0.0, 5.0],   # Straight, slow
        [0.0, 10.0],  # Straight, medium
        [0.0, 15.0],  # Straight, fast
        [0.1, 10.0],  # Right turn, medium
        [-0.1, 10.0],  # Left turn, medium
    ]

    for i, action in enumerate(test_actions):
        print(f"\n--- Testing action {action} ---")
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0

        # Apply same action for multiple steps
        for _ in range(20):
            step += 1
            obs, reward, done, truncated, info = env.step(np.array([[action[0], action[1]]]))
            done = done or truncated
            total_reward += reward

            if done:
                print(f"  Step {step}: TERMINATED with reward {reward:.4f}")
                break
            elif step % 5 == 0:
                print(f"  Step {step}: reward={reward:.4f}, total={total_reward:.4f}")

        if not done:
            print(f"  Completed 20 steps without termination, total reward: {total_reward:.4f}")

    env.close()


if __name__ == "__main__":
    analyze_reward_spike()
    compare_reward_functions()
    test_static_actions()
