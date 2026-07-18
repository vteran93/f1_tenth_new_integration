#!/usr/bin/env python3
"""
F1TENTH RL Integration - Quick Demo

This script demonstrates the advanced features of your F1TENTH RL integration.
"""

import sys
from pathlib import Path

# Use local f1tenth_benchmarks module
import f1tenth_benchmarks


def demo_config_manager():
    """Demo the configuration manager."""
    print("🔧 Configuration Manager Demo")
    print("=" * 40)

    from config_manager import ConfigurationManager, Algorithm, Environment

    # Create manager and generate templates
    manager = ConfigurationManager()
    templates = manager.generate_config_templates()

    print(f"✓ Generated {len(templates)} configuration templates:")
    for template in templates:
        print(f"  - {template.name}: {template.description}")

    # Create a custom configuration
    custom_config = manager.create_experiment_config(
        name="demo_experiment",
        algorithm=Algorithm.PPO,
        environment=Environment.OVAL_SMALL,
        description="Demo configuration for quick testing"
    )

    # Modify some settings
    custom_config.training.timesteps_total = 100_000  # Quick training
    custom_config.environment.num_agents = 2
    if custom_config.ppo:  # Check if PPO config exists
        custom_config.ppo.lr = 0.0001

    # Save configuration
    config_path = manager.save_config(custom_config)
    print(f"✓ Custom configuration saved to: {config_path.name}")

    # Validate
    issues = manager.validate_config(custom_config)
    if not issues:
        print("✓ Configuration validation passed")

    return str(config_path)


def demo_quick_training():
    """Demo the quick training capabilities."""
    print("\n🏃‍♂️ Quick Training Demo")
    print("=" * 40)

    print("Quick training options available:")
    print("  1. python quick_train.py --mode quick --timesteps 50000")
    print("  2. python quick_train.py --mode interactive")
    print("  3. python quick_train.py --mode config --config your_config.yaml")

    print("\nTraining features:")
    print("  ✓ Automatic environment registration")
    print("  ✓ Built-in checkpointing")
    print("  ✓ Progress monitoring")
    print("  ✓ Resource optimization")
    print("  ✓ Multiple algorithms (PPO, SAC)")


def demo_evaluation_system():
    """Demo the evaluation system."""
    print("\n🧪 Model Evaluation Demo")
    print("=" * 40)

    from model_evaluator import ModelEvaluator, EvaluationConfig

    # Create evaluation configuration
    config = EvaluationConfig(
        num_episodes=10,  # Quick demo
        test_maps=["oval_small"],
        test_noise_levels=[0.0, 0.01],
        output_dir=Path("./demo_evaluation_results")
    )

    evaluator = ModelEvaluator(config)
    print("✓ Model evaluator configured")

    print("\nEvaluation capabilities:")
    print("  ✓ Multi-map testing")
    print("  ✓ Robustness analysis (noise tolerance)")
    print("  ✓ Comprehensive metrics collection")
    print("  ✓ Model comparison")
    print("  ✓ Automated report generation")

    # Test environment creation
    env = evaluator.create_environment("oval_small")
    obs, info = env.reset()
    env.close()
    print("✓ Evaluation environment ready")


def demo_dashboard():
    """Demo the dashboard capabilities."""
    print("\n📊 Analysis Dashboard Demo")
    print("=" * 40)

    print("Dashboard features:")
    print("  ✓ Interactive training metrics visualization")
    print("  ✓ Benchmark data analysis")
    print("  ✓ Model performance comparison")
    print("  ✓ Real-time monitoring capabilities")
    print("  ✓ Professional-grade plots with Plotly")

    print("\nTo start the dashboard:")
    print("  streamlit run analysis_dashboard.py")
    print("  → Opens in browser at http://localhost:8501")


def demo_multiagent():
    """Demo multiagent capabilities."""
    print("\n👥 Multiagent Integration Demo")
    print("=" * 40)

    # Test multiagent environment
    import gymnasium as gym

    env = gym.make('f1tenth_gym:f1tenth-v0', config={
        'map': 'oval_small',
        'num_agents': 4,
        'timestep': 0.01
    })

    obs, info = env.reset()
    print(f"✓ 4-agent environment created")
    print(f"✓ Observation keys: {list(obs.keys())}")

    # Test callbacks
    from examples.multiagent.callbacks import BenchmarkDataCollectorCallback
    callback = BenchmarkDataCollectorCallback()
    print("✓ Benchmark data collection ready")

    env.close()


def main():
    """Run the complete demo."""
    print("🏎️ F1TENTH RL Advanced Features Demo")
    print("=" * 50)

    try:
        # Demo each component
        config_path = demo_config_manager()
        demo_quick_training()
        demo_evaluation_system()
        demo_dashboard()
        demo_multiagent()

        print("\n🎯 Next Steps")
        print("=" * 40)
        print("1. Try quick training:")
        print(f"   python quick_train.py --mode config --config {config_path}")

        print("\n2. Start the analysis dashboard:")
        print("   streamlit run analysis_dashboard.py")

        print("\n3. Run comprehensive tests:")
        print("   python test_advanced_features.py")

        print("\n4. Explore multiagent training:")
        print("   cd examples/multiagent && python run.py")

        print("\n🚀 Your F1TENTH RL environment is ready for advanced research!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
