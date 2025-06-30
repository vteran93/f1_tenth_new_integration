#!/usr/bin/env python3
"""
Multi-Training Rewards Test Script

This script tests all reward functions from both rewards.py and rewards_pepe.py
with both PPO and SAC algorithms. It automatically creates configurations and
runs training sessions with different timestep counts to validate functionality.

Usage:
    python multi_training_rewards.py
    python multi_training_rewards.py --timesteps 5000
    python multi_training_rewards.py --algorithms PPO
    python multi_training_rewards.py --skip_pepe
"""

import argparse
import os
import sys
import importlib
import tempfile
import yaml
from lib.utils import load_config, init_ray, get_logger, suppress_warnings
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec
import datetime

# Setup logging and suppress warnings
suppress_warnings()
logger = get_logger(__name__)

# Algorithm configuration mapping
ALGO_MAP = {
    "PPO": (PPOConfig, "ppo_config"),
    "SAC": (SACConfig, "sac_config"),
}

# Reward classes from rewards.py (inheriting from MultiAgentF110)
REWARDS_PY_CLASSES = [
    "ProgressRewardEnv",
    "SpeedRewardEnv"
]

# Reward classes from rewards.py using BaseReward pattern (used with MultiAgentF110)
REWARDS_PY_BASE_CLASSES = [
    "SACBasicReward",
    "SACGeminiReward",
    "SpeedReward",
    "SafetyReward"
]

# Reward classes from rewards_pepe.py (inheriting from RewardFunction and MultiAgentF110)
REWARDS_PEPE_CLASSES = [
    "GeminiReward",
    "SpeedReward",
    "WaypointReward",
    "CompetitiveOvertakingReward"
]


def create_temp_config(algorithm, reward_function, timesteps, base_config_path):
    """
    Create a temporary configuration file for training.

    Args:
        algorithm (str): Algorithm name (PPO or SAC)
        reward_function (str): Reward function class name
        timesteps (int): Total timesteps for training
        base_config_path (str): Path to base configuration file

    Returns:
        str: Path to temporary configuration file
    """
    # Load base configuration
    if algorithm == "PPO":
        base_config = load_config("configs/ppo_config.yaml")
    else:  # SAC
        base_config = load_config("configs/sac_config.yaml")

    # Update configuration for this test
    base_config['experiment_name'] = f"test_{algorithm}_{reward_function}_{timesteps}"
    base_config['training']['algorithm'] = algorithm
    base_config['training']['reward_function'] = reward_function
    base_config['training']['timesteps_total'] = timesteps
    base_config['training']['eval_interval'] = max(timesteps // 10, 100)  # Eval every 10% or min 100 steps

    # Reduce some parameters for faster testing
    base_config['env_config']['num_agents'] = 2

    # Adjust algorithm-specific settings for faster testing
    if algorithm == "PPO":
        base_config['ppo_config']['train_batch_size'] = min(1000, timesteps // 4)
        base_config['ppo_config']['lr'] = 0.0001  # Slightly higher for faster learning
    else:  # SAC
        # Keep SAC settings but ensure they work with low timesteps
        pass

    # Create temporary config file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix=f'{algorithm}_{reward_function}_')
    try:
        with os.fdopen(temp_fd, 'w') as tmp_file:
            yaml.dump(base_config, tmp_file, default_flow_style=False)
    except Exception:
        os.close(temp_fd)
        raise

    return temp_path


def get_reward_class(config):
    """Get reward class from either rewards.py or rewards_pepe.py"""
    reward_function_name = config['training']['reward_function']

    # Try rewards.py first
    try:
        reward_module = importlib.import_module('lib.rewards')
        return getattr(reward_module, reward_function_name)
    except AttributeError:
        pass

    # Try rewards_pepe.py
    try:
        reward_module = importlib.import_module('lib.rewards_pepe')
        return getattr(reward_module, reward_function_name)
    except AttributeError:
        raise ValueError(f"Reward function '{reward_function_name}' not found in rewards.py or rewards_pepe.py")


def get_algorithm_config(config, env_config, policies):
    """Create algorithm configuration"""
    algorithm_name = config['training']['algorithm']

    if algorithm_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    AlgoConfigClass, config_key = ALGO_MAP[algorithm_name]

    # Get algorithm config from the merged config
    algo_config_file = config.get(config_key, {}).copy()
    env_kwargs = algo_config_file.get('environment', {})

    # Clean up the config before passing to training
    if 'environment' in algo_config_file:
        del algo_config_file['environment']

    algo_config = (
        AlgoConfigClass()
        .environment(get_reward_class(config), env_config=env_config, **env_kwargs)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=0,  # Start with 0 for testing
            num_envs_per_env_runner=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .evaluation(
            evaluation_interval=config["training"]["eval_interval"],
            evaluation_num_env_runners=1,
            evaluation_config={"seed": 42},
        )
        .debugging(seed=42)
    )

    algo_config.training(**algo_config_file)
    return algo_config


def create_env(config, render_mode=None):
    """Create environment instance"""
    env_config = config['env_config'].copy()

    if render_mode:
        env_config["render_mode"] = render_mode

    reward_class = get_reward_class(config)
    return reward_class(env_config=env_config), env_config


def test_reward_function(algorithm, reward_function, timesteps):
    """
    Test a single reward function with the specified algorithm.

    Args:
        algorithm (str): Algorithm name (PPO or SAC)
        reward_function (str): Reward function class name
        timesteps (int): Total timesteps for training

    Returns:
        bool: True if test passed, False otherwise
    """
    logger.info(f"=== Testing {algorithm} with {reward_function} ({timesteps} timesteps) ===")

    try:
        # Create temporary configuration
        temp_config_path = create_temp_config(algorithm, reward_function, timesteps, None)

        try:
            # Load configuration
            config = load_config(temp_config_path)

            # Create environment to validate reward function
            temp_env, env_config = create_env(config)

            # Check if we should use shared policy
            shared_policy = config['training'].get('shared_policy', True)

            if shared_policy:
                logger.info(f"Using shared policy for {reward_function}")
                policy_learn = PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
                policies = {agent: policy_learn for agent in temp_env.agents}
            else:
                logger.info(f"Using individual policies for {reward_function}")
                policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
                            for agent in temp_env.agents}

            temp_env.close()

            # Get algorithm configuration
            config_algo = get_algorithm_config(config, env_config, policies)

            # Set up storage path for this test
            test_storage_path = f"../models_test/{algorithm}_{reward_function}_{timesteps}"
            os.makedirs(test_storage_path, exist_ok=True)

            # Run training
            tune.run(
                algorithm,
                config=config_algo.to_dict(),
                stop={"timesteps_total": timesteps},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_score_attribute="episode_reward_mean",
                    checkpoint_score_order="max",
                    num_to_keep=1,  # Keep only 1 checkpoint for testing
                    checkpoint_at_end=True,
                    checkpoint_frequency=max(timesteps // 5, 100)  # Checkpoint every 20% or min 100 steps
                ),
                storage_path=test_storage_path,
                name=f"test_{reward_function}",
                trial_name_creator=lambda trial: f"{algorithm}_{reward_function}_{datetime.datetime.now().strftime('%H%M%S')}",
                verbose=1,  # Reduce verbosity for testing
            )

            logger.info(f"✅ SUCCESS: {algorithm} + {reward_function} ({timesteps} timesteps)")
            return True

        finally:
            # Clean up temporary config file
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)

    except Exception as e:
        logger.error(f"❌ FAILED: {algorithm} + {reward_function} ({timesteps} timesteps) - {str(e)}")
        return False


def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description="Test all reward functions with PPO and SAC")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of timesteps for each test (default: 1000)")
    parser.add_argument("--algorithms", nargs='+', choices=["PPO", "SAC"], default=["PPO", "SAC"],
                        help="Algorithms to test (default: both PPO and SAC)")
    parser.add_argument("--skip_pepe", action="store_true",
                        help="Skip testing rewards_pepe.py classes")
    parser.add_argument("--skip_base", action="store_true",
                        help="Skip testing BaseReward classes from rewards.py")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests with minimal timesteps (100 steps)")

    args = parser.parse_args()

    if args.quick:
        args.timesteps = 100
        logger.info("Quick mode: Using 100 timesteps per test")

    # Initialize Ray
    init_ray()

    # Collect all reward functions to test
    reward_functions = []

    # Add rewards.py classes (MultiAgentF110 based)
    for reward_func in REWARDS_PY_CLASSES:
        reward_functions.append((reward_func, "rewards.py", "MultiAgentF110"))

    # Add rewards.py BaseReward classes (if not skipped)
    if not args.skip_base:
        for reward_func in REWARDS_PY_BASE_CLASSES:
            reward_functions.append((reward_func, "rewards.py", "BaseReward"))

    # Add rewards_pepe.py classes (if not skipped)
    if not args.skip_pepe:
        for reward_func in REWARDS_PEPE_CLASSES:
            reward_functions.append((reward_func, "rewards_pepe.py", "RewardFunction"))

    # Test all combinations
    total_tests = len(reward_functions) * len(args.algorithms)
    passed_tests = 0
    failed_tests = []

    logger.info(f"Starting {total_tests} tests with {args.timesteps} timesteps each")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"Reward functions: {[rf[0] for rf in reward_functions]}")

    for algorithm in args.algorithms:
        for reward_func, source_file, base_class in reward_functions:
            logger.info(f"Testing {reward_func} from {source_file} (inherits from {base_class})")

            success = test_reward_function(algorithm, reward_func, args.timesteps)

            if success:
                passed_tests += 1
            else:
                failed_tests.append(f"{algorithm} + {reward_func}")

    # Print summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {len(failed_tests)}")

    if failed_tests:
        logger.info("Failed tests:")
        for failed_test in failed_tests:
            logger.info(f"  - {failed_test}")
    else:
        logger.info("🎉 All tests passed!")

    logger.info("=" * 60)

    # Return appropriate exit code
    return 0 if len(failed_tests) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
