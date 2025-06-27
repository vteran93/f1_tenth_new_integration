#!/usr/bin/env python3
"""
Basic test to validate our implementation structure.
"""


def test_imports():
    """Test basic imports."""
    print("🔍 Testing imports...")

    try:
        # Test reward system import
        from rewards import get_reward_function, REWARD_FUNCTIONS
        print("✅ Rewards module imported successfully")

        # Test reward functions
        ppo_reward = get_reward_function("ppo", "progress")
        sac_reward = get_reward_function("sac", "gemini")
        print(f"✅ Reward functions: {ppo_reward.__class__.__name__}, {sac_reward.__class__.__name__}")

        # Test reward dictionary structure
        print(f"✅ Available algorithms: {list(REWARD_FUNCTIONS.keys())}")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_track_files():
    """Test track files exist."""
    print("\n🛣️  Testing track files...")

    import os
    tracks_dir = "./tracks"

    if not os.path.exists(tracks_dir):
        print("❌ Tracks directory not found")
        return False

    tracks = [d for d in os.listdir(tracks_dir) if os.path.isdir(os.path.join(tracks_dir, d))]
    print(f"📁 Found tracks: {tracks}")

    # Check each track has required files
    complete_tracks = 0
    for track in tracks:
        png_file = f"{tracks_dir}/{track}/{track}_map.png"
        yaml_file = f"{tracks_dir}/{track}/{track}_map.yaml"

        if os.path.exists(png_file) and os.path.exists(yaml_file):
            print(f"  ✅ {track}: Complete")
            complete_tracks += 1
        else:
            print(f"  ❌ {track}: Missing files")

    print(f"📊 {complete_tracks}/{len(tracks)} tracks are complete")
    return complete_tracks > 0


def test_file_structure():
    """Test our file structure is correct."""
    print("\n📁 Testing file structure...")

    import os

    files_to_check = [
        "multiagent_env.py",
        "rewards.py",
        "training_with_oval_track.py",
        "create_tracks.py",
    ]

    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_exist = False

    return all_exist


def main():
    """Run basic tests."""
    print("🧪 Basic System Validation")
    print("=" * 30)

    tests = [
        ("File Structure", test_file_structure),
        ("Track Files", test_track_files),
        ("Imports", test_imports),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 Results:")
    print("-" * 15)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Score: {passed}/{len(tests)}")

    if passed == len(tests):
        print("🎉 Basic validation passed!")
        print("\n💡 Next steps:")
        print("  1. Test environment creation (may need f1tenth_gym)")
        print("  2. Run training: python training_with_oval_track.py --train --algo ppo --track oval_small")
    else:
        print("⚠️  Some basic tests failed")


if __name__ == "__main__":
    main()
