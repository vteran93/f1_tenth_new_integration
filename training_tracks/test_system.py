#!/usr/bin/env python3
"""
Simple test script to validate our multiagent environment and reward system.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from multiagent_env import MultiAgentF110PPO, MultiAgentF110SAC
    from rewards import get_reward_function, list_available_rewards
    print("✅ Successfully imported environment classes")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_reward_system():
    """Test the polymorphic reward system."""
    print("\n🏆 Testing Reward System:")
    print("-" * 30)

    try:
        # Test PPO rewards
        ppo_progress = get_reward_function("ppo", "progress")
        ppo_speed = get_reward_function("ppo", "speed")
        ppo_safety = get_reward_function("ppo", "safety")

        print(f"✅ PPO Progress: {ppo_progress.__class__.__name__}")
        print(f"✅ PPO Speed: {ppo_speed.__class__.__name__}")
        print(f"✅ PPO Safety: {ppo_safety.__class__.__name__}")

        # Test SAC rewards
        sac_basic = get_reward_function("sac", "basic")
        sac_gemini = get_reward_function("sac", "gemini")
        sac_speed = get_reward_function("sac", "speed")

        print(f"✅ SAC Basic: {sac_basic.__class__.__name__}")
        print(f"✅ SAC Gemini: {sac_gemini.__class__.__name__}")
        print(f"✅ SAC Speed: {sac_speed.__class__.__name__}")

        return True

    except Exception as e:
        print(f"❌ Reward system error: {e}")
        return False


def test_environment_creation():
    """Test creation of environment classes."""
    print("\n🏁 Testing Environment Creation:")
    print("-" * 35)

    try:
        # Simple config
        env_config = {
            "map": "Spielberg",  # Use default map
            "num_agents": 2,
            "timestep": 0.01,
            "num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "original"},
            "reset_config": {"type": "cl_grid_static"},
        }

        # Test PPO environment
        print("Creating PPO environment...")
        ppo_reward = get_reward_function("ppo", "progress")
        ppo_env = MultiAgentF110PPO(env_config, reward_function=ppo_reward)
        print(f"✅ PPO Environment: {len(ppo_env.agents)} agents")
        print(f"   Action space: {type(ppo_env.action_space)}")
        print(f"   Observation space: {type(ppo_env.observation_space)}")
        ppo_env.close()

        # Test SAC environment
        print("Creating SAC environment...")
        sac_reward = get_reward_function("sac", "gemini")
        env_config["reset_config"] = {"type": "rl_random_static"}  # SAC config
        sac_env = MultiAgentF110SAC(env_config, reward_function=sac_reward)
        print(f"✅ SAC Environment: {len(sac_env.agents)} agents")
        print(f"   Action space: {type(sac_env.action_space)}")
        print(f"   Observation space: {type(sac_env.observation_space)}")
        sac_env.close()

        return True

    except Exception as e:
        print(f"❌ Environment creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_track_loading():
    """Test loading of custom tracks."""
    print("\n🛣️  Testing Track Loading:")
    print("-" * 28)

    # Check if tracks directory exists
    tracks_dir = "./tracks"
    if not os.path.exists(tracks_dir):
        print("❌ No tracks directory found")
        return False

    # List available tracks
    tracks = [d for d in os.listdir(tracks_dir) if os.path.isdir(os.path.join(tracks_dir, d))]
    print(f"📁 Found {len(tracks)} tracks:")

    for track in tracks:
        track_path = os.path.join(tracks_dir, track)
        png_file = os.path.join(track_path, f"{track}_map.png")
        yaml_file = os.path.join(track_path, f"{track}_map.yaml")

        if os.path.exists(png_file) and os.path.exists(yaml_file):
            print(f"  ✅ {track} - Complete")
        else:
            print(f"  ❌ {track} - Missing files")

    return len(tracks) > 0


def test_polymorphism():
    """Test polymorphic behavior of reward functions."""
    print("\n🔄 Testing Polymorphism:")
    print("-" * 25)

    try:
        # Create different reward instances
        rewards = [
            ("PPO Progress", get_reward_function("ppo", "progress")),
            ("SAC Gemini", get_reward_function("sac", "gemini")),
            ("Shared Speed", get_reward_function("shared", "speed")),
        ]

        # Test polymorphic behavior
        for name, reward_obj in rewards:
            # Check if it has the required method
            if hasattr(reward_obj, '_get_rewards'):
                print(f"✅ {name}: implements _get_rewards()")
            else:
                print(f"❌ {name}: missing _get_rewards()")
                return False

        print("✅ All reward functions implement required interface")
        return True

    except Exception as e:
        print(f"❌ Polymorphism test error: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 F1TENTH Multi-Agent Environment Test Suite")
    print("=" * 50)

    tests = [
        ("Reward System", test_reward_system),
        ("Environment Creation", test_environment_creation),
        ("Track Loading", test_track_loading),
        ("Polymorphism", test_polymorphism),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 Test Summary:")
    print("=" * 20)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! System is ready for training.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
