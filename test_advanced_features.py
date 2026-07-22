#!/usr/bin/env python3
"""
Comprehensive Integration Test for F1TENTH RL Advanced Features

This script tests all components of the advanced F1TENTH RL integration including:
- Basic environment functionality
- Configuration management
- Training capabilities
- Evaluation system
- Dashboard components
"""

import sys
from pathlib import Path
import logging
import tempfile
import shutil

# Use local f1tenth_benchmarks module
import f1tenth_benchmarks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_integration():
    """Test basic f1tenth_gym and f1tenth_benchmarks integration."""
    print("\n🔧 Testing Basic Integration...")

    try:
        # Test imports
        import f1tenth_gym
        import gymnasium as gym
        import ray
        import numpy as np
        import pandas as pd
        print("✓ Core libraries imported successfully")

        # Test f1tenth_benchmarks access
        import f1tenth_benchmarks
        print("✓ f1tenth_benchmarks imported successfully")

        # Test environment creation
        env = gym.make('f1tenth_gym:f1tenth-v0', config={
            'map': 'oval_small',
            'num_agents': 2,
            'timestep': 0.01,
            'integrator': 'rk4'
        })

        obs, info = env.reset()
        print("✓ F1TENTH environment created and reset successfully")
        env.close()

        return True

    except Exception as e:
        print(f"✗ Basic integration test failed: {e}")
        return False


def test_config_manager():
    """Test the configuration management system."""
    print("\n⚙️ Testing Configuration Manager...")

    try:
        from config_manager import ConfigurationManager, Algorithm, Environment

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigurationManager(config_dir=Path(temp_dir))

            # Test configuration creation
            config = manager.create_experiment_config(
                name="test_experiment",
                algorithm=Algorithm.PPO,
                environment=Environment.OVAL_SMALL
            )
            print("✓ Configuration created successfully")

            # Test configuration saving/loading
            config_path = manager.save_config(config)
            loaded_config = manager.load_config(config_path)
            print("✓ Configuration save/load works")

            # Test validation
            issues = manager.validate_config(config)
            if not issues:
                print("✓ Configuration validation passed")
            else:
                print(f"⚠ Configuration has issues: {issues}")

            # Test template generation
            templates = manager.generate_config_templates()
            print(f"✓ Generated {len(templates)} configuration templates")

        return True

    except Exception as e:
        print(f"✗ Configuration manager test failed: {e}")
        return False


def test_model_evaluator():
    """Test the model evaluation system."""
    print("\n🧪 Testing Model Evaluator...")

    try:
        from model_evaluator import ModelEvaluator, EvaluationConfig, EvaluationMetrics

        # Create test configuration
        config = EvaluationConfig(
            num_episodes=2,  # Minimal for testing
            test_maps=["oval_small"],
            test_noise_levels=[0.0],
            output_dir=Path("./test_evaluation_results")
        )

        evaluator = ModelEvaluator(config)
        print("✓ Model evaluator created successfully")

        # Test environment creation
        env = evaluator.create_environment("oval_small")
        obs, info = env.reset()
        env.close()
        print("✓ Evaluation environment creation works")

        # Test metrics structure
        metrics = EvaluationMetrics()
        metrics.lap_times = [10.0, 12.0, 11.5]
        metrics.completion_rates = [1.0, 1.0, 0.8]
        metrics_dict = metrics.to_dict()
        print("✓ Metrics collection and conversion works")

        # Cleanup test directory
        if Path("./test_evaluation_results").exists():
            shutil.rmtree("./test_evaluation_results")

        return True

    except Exception as e:
        print(f"✗ Model evaluator test failed: {e}")
        return False


def test_quick_training():
    """Test the quick training script functionality."""
    print("\n🏃‍♂️ Testing Quick Training Script...")

    try:
        # Import training components
        from quick_train import (
            create_f1tenth_env,
            setup_ppo_config,
            create_default_config,
            load_experiment_config
        )

        # Test default configuration creation
        config = create_default_config()
        print("✓ Default configuration created")

        # Test environment factory
        env_config = {
            'map': 'oval_small',
            'num_agents': 1,
            'timestep': 0.01,
            'integrator': 'rk4'
        }

        # Create mock env context
        class MockEnvContext:
            def __init__(self, config):
                self._config = config

            def get(self, key, default=None):
                return self._config.get(key, default)

        mock_context = MockEnvContext(env_config)
        env = create_f1tenth_env(mock_context)
        obs, info = env.reset()
        env.close()
        print("✓ Environment factory works")

        # Test algorithm configuration
        ppo_config = setup_ppo_config(env_config, config['training'])
        print("✓ PPO configuration setup works")

        return True

    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        return False


def test_dashboard_components():
    """Test dashboard components (without actually starting Streamlit)."""
    print("\n📊 Testing Dashboard Components...")

    try:
        # Test dashboard imports
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        print("✓ Dashboard dependencies available")

        # Test dashboard class import
        from analysis_dashboard import F1TenthAnalysisDashboard

        dashboard = F1TenthAnalysisDashboard()
        print("✓ Dashboard class instantiated")

        # Test data loading methods (with non-existent paths - should handle gracefully)
        result = dashboard.load_training_results("./non_existent_path")
        if result is None:
            print("✓ Training results loading handles missing files gracefully")

        benchmark_result = dashboard.load_benchmark_data("non_existent_experiment")
        if benchmark_result is None:
            print("✓ Benchmark data loading handles missing files gracefully")

        return True

    except Exception as e:
        print(f"✗ Dashboard components test failed: {e}")
        return False


def test_multiagent_integration():
    """Test multiagent training components."""
    print("\n👥 Testing Multiagent Integration...")

    try:
        # Test multiagent callbacks import
        from examples.multiagent.callbacks import BenchmarkDataCollectorCallback, BenchmarkMetricsSaver

        # Create callback instances
        data_collector = BenchmarkDataCollectorCallback()
        print("✓ BenchmarkDataCollectorCallback created")

        metrics_saver = BenchmarkMetricsSaver({'name': 'test', 'storage_path': './test'})
        print("✓ BenchmarkMetricsSaver created")

        # Test multiagent environment
        import gymnasium as gym
        env = gym.make('f1tenth_gym:f1tenth-v0', config={
            'map': 'oval_small',
            'num_agents': 4,  # Multi-agent
            'timestep': 0.01
        })

        obs, info = env.reset()
        print("✓ Multi-agent environment works")

        # Test that observation is a dict (multi-agent format)
        if isinstance(obs, dict):
            print(f"✓ Multi-agent observations format correct (keys: {list(obs.keys())})")

        env.close()

        return True

    except Exception as e:
        print(f"✗ Multiagent integration test failed: {e}")
        return False


def test_ray_integration():
    """Test Ray RLLib integration."""
    print("\n⚡ Testing Ray Integration...")

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray import tune

        # Initialize Ray (ignore if already initialized)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, local_mode=True)

        print("✓ Ray initialized successfully")

        # Test environment registration
        from quick_train import create_f1tenth_env
        tune.register_env('test_f1tenth_env', create_f1tenth_env)
        print("✓ Environment registration works")

        # Test basic PPO config creation
        config = (PPOConfig()
                  .environment(env='test_f1tenth_env',
                               env_config={'map': 'oval_small', 'num_agents': 1})
                  .framework('torch')
                  .training(train_batch_size=100, gamma=0.99)
                  .env_runners(num_env_runners=0))  # No workers for testing

        print("✓ PPO configuration created")

        return True

    except Exception as e:
        print(f"✗ Ray integration test failed: {e}")
        return False
    finally:
        # Clean shutdown
        try:
            ray.shutdown()
        except:
            pass


def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("🧪 F1TENTH RL Advanced Features - Comprehensive Test")
    print("=" * 60)

    tests = [
        ("Basic Integration", test_basic_integration),
        ("Configuration Manager", test_config_manager),
        ("Model Evaluator", test_model_evaluator),
        ("Quick Training", test_quick_training),
        ("Dashboard Components", test_dashboard_components),
        ("Multiagent Integration", test_multiagent_integration),
        ("Ray Integration", test_ray_integration),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Your F1TENTH RL integration is ready.")
        print("\n🚀 Quick start commands:")
        print("   python quick_train.py --mode quick --timesteps 50000")
        print("   streamlit run analysis_dashboard.py")
        print("   python -c \"from config_manager import *; ConfigurationManager().generate_config_templates()\"")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
