#!/usr/bin/env python3

"""
Direct testing of reset functions to understand the broken reset system.
This script tests the reset functions directly without environment wrapper.
"""

from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.reset import make_reset_fn
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import f1tenth_gym
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import reset functions


def test_reset_function_direct():
    """Test reset functions directly without environment wrapper."""
    print("=== TESTING RESET FUNCTIONS DIRECTLY ===")

    # Load track
    try:
        track = Track.from_track_name("Spielberg")
        print(f"Track loaded: {track.spec.name}")
        print(f"Track waypoints: {len(track.raceline.xs)}")
        print(f"Track origin: {track.spec.origin}")
        print(f"Track resolution: {track.spec.resolution}")
    except Exception as e:
        print(f"Failed to load track: {e}")
        return

    # Test different reset configurations
    reset_configs = [
        {"type": "rl_random_static"},
        {"type": "rl_random_random"},
        {"type": "rl_grid_static"}
    ]

    num_agents = 1

    for config in reset_configs:
        print(f"\n--- Testing reset config: {config} ---")
        try:
            # Create reset function
            reset_fn = make_reset_fn(
                track=track,
                num_agents=num_agents,
                **config
            )
            print(f"Reset function created: {type(reset_fn).__name__}")

            # Sample poses multiple times
            for i in range(5):
                poses = reset_fn.sample()
                print(f"Sample {i+1}: shape={poses.shape}, poses={poses}")

                # Check if all poses are the same
                if i > 0 and np.allclose(poses, first_poses):
                    print("  ⚠️  IDENTICAL TO FIRST SAMPLE!")
                elif i == 0:
                    first_poses = poses.copy()
                else:
                    print("  ✓  Different from first sample")

        except Exception as e:
            print(f"Error with config {config}: {e}")
            import traceback
            traceback.print_exc()


def test_reset_function_components():
    """Test individual components of the reset system."""
    print("\n=== TESTING RESET SYSTEM COMPONENTS ===")

    # Load track
    track = Track.from_track_name("Spielberg")

    # Test waypoint sampling directly
    print("\n--- Testing waypoint sampling ---")
    from f1tenth_gym.envs.reset.utils import sample_around_waypoint

    try:
        # Test with different waypoint IDs
        for waypoint_id in [0, 10, 50]:
            poses = sample_around_waypoint(
                reference_line=track.raceline,
                waypoint_id=waypoint_id,
                n_agents=1,
                min_dist=1.5,
                max_dist=2.5,
                move_laterally=True
            )
            print(f"Waypoint {waypoint_id}: {poses}")

    except Exception as e:
        print(f"Error in waypoint sampling: {e}")
        import traceback
        traceback.print_exc()

    # Test pose sampling around a fixed point
    print("\n--- Testing pose sampling around fixed point ---")
    from f1tenth_gym.envs.reset.utils import sample_around_pose

    try:
        base_pose = np.array([1.0, 2.0, 0.5])  # Fixed pose
        for i in range(3):
            poses = sample_around_pose(
                pose=base_pose,
                n_agents=1,
                min_dist=0.5,
                max_dist=1.0
            )
            print(f"Sample {i+1} around {base_pose}: {poses}")

    except Exception as e:
        print(f"Error in pose sampling: {e}")
        import traceback
        traceback.print_exc()


def test_track_raceline():
    """Test track raceline data."""
    print("\n=== TESTING TRACK RACELINE DATA ===")

    track = Track.from_track_name("Spielberg")
    raceline = track.raceline

    print(f"Raceline points: {raceline.n}")
    print(f"First 5 x-coordinates: {raceline.xs[:5]}")
    print(f"First 5 y-coordinates: {raceline.ys[:5]}")
    print(f"X range: [{np.min(raceline.xs):.3f}, {np.max(raceline.xs):.3f}]")
    print(f"Y range: [{np.min(raceline.ys):.3f}, {np.max(raceline.ys):.3f}]")

    # Test manual waypoint access
    print(f"\nWaypoint 0: ({raceline.xs[0]:.3f}, {raceline.ys[0]:.3f})")
    print(f"Waypoint 10: ({raceline.xs[10]:.3f}, {raceline.ys[10]:.3f})")
    print(f"Waypoint 50: ({raceline.xs[50]:.3f}, {raceline.ys[50]:.3f})")


if __name__ == "__main__":
    print("Testing F1TENTH reset system components...")

    test_track_raceline()
    test_reset_function_components()
    test_reset_function_direct()

    print("\nDirect reset function testing completed!")
