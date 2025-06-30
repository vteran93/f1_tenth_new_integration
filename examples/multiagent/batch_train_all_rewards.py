#!/usr/bin/env python3
"""
Batch Training Script for All Reward Functions

This script trains models with all reward functions from both rewards.py and rewards_pepe.py
using both PPO and SAC algorithms with 5000 timesteps each. The goal is to generate
trained models for comparison and analysis.

Based on run.py logic but designed for batch processing multiple reward functions.

Usage:
    python batch_train_all_rewards.py
    python batch_train_all_rewards.py --algorithms PPO
    python batch_train_all_rewards.py --timesteps 10000
    python batch_train_all_rewards.py --skip_pepe
"""

import argparse
import importlib
import os
import sys
from lib.utils import load_config, init_ray, get_logger, suppress_warnings
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec
import datetime

suppress_warnings()
logger = get_logger(__name__)

ALGO_MAP = {
    "PPO": (PPOConfig, "ppo_config"),
    "SAC": (SACConfig, "sac_config"),
}

# Reward classes from rewards.py (inheriting from MultiAgentF110)
REWARDS_PY_CLASSES = [
    "ProgressRewardEnv",
    "SpeedRewardEnv"
]

# Reward classes from rewards.py using BaseReward pattern
REWARDS_PY_BASE_CLASSES = [
    "SACBasicReward",
    "SACGeminiReward",
    "SpeedReward",
    "SafetyReward"
]

# Reward classes from rewards_pepe.py
REWARDS_PEPE_CLASSES = [
    "GeminiReward",
    "SpeedReward",
    "WaypointReward",
    "CompetitiveOvertakingReward"
]


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
    """Create algorithm configuration - Based on run.py"""
    algorithm_name = config['training']['algorithm']

    if algorithm_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    AlgoConfigClass, config_key = ALGO_MAP[algorithm_name]

    # Get algorithm config from the merged config instead of loading separate file
    algo_config_file = config.get(config_key, {}).copy()  # Make a copy to avoid mutations
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
            num_env_runners=0,
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
        ).env_runners(
            num_env_runners=5,            # 4 procesos en paralelo
            num_envs_per_env_runner=10,    # 4 entornos vectorizados por proceso
            gym_env_vectorize_mode="ASYNC"
        )
        .debugging(seed=42)
    )

    algo_config.training(**algo_config_file)
    return algo_config


def create_env(config, render_mode=None):
    """Loads environment config and creates an environment instance - Based on run.py"""
    # Since environment is always included in the same config, use embedded env_config
    env_config = config['env_config'].copy()

    if render_mode:
        env_config["render_mode"] = render_mode

    reward_class = get_reward_class(config)
    return reward_class(env_config=env_config), env_config


def create_training_config(algorithm, reward_function, timesteps, base_storage_path):
    """
    Create a training configuration for a specific reward function.

    Args:
        algorithm (str): Algorithm name (PPO or SAC)
        reward_function (str): Reward function class name
        timesteps (int): Total timesteps for training
        base_storage_path (str): Base path for storing models

    Returns:
        dict: Training configuration
    """
    # Load base configuration
    if algorithm == "PPO":
        base_config = load_config("configs/ppo_config.yaml")
    else:  # SAC
        base_config = load_config("configs/sac_config.yaml")

    # Update configuration for this training
    experiment_name = f"batch_training_{algorithm}_{reward_function}"
    base_config['experiment_name'] = experiment_name
    base_config['training']['algorithm'] = algorithm
    base_config['training']['reward_function'] = reward_function
    base_config['training']['timesteps_total'] = timesteps
    base_config['training']['eval_interval'] = max(timesteps // 20, 250)  # Eval every 5% or min 250 steps

    # Set storage path (convert to absolute path to avoid URI issues)
    storage_path = os.path.join(base_storage_path, experiment_name)
    base_config['storage_path'] = os.path.abspath(storage_path)

    return base_config


def run_training_batch(config):
    """Run training for a single configuration - Based on run.py run_training function"""
    logger.info(f"Starting training: {config['experiment_name']}")
    logger.info(f"Algorithm: {config['training']['algorithm']}")
    logger.info(f"Reward Function: {config['training']['reward_function']}")
    logger.info(f"Timesteps: {config['training']['timesteps_total']}")

    temp_env, env_config = create_env(config)

    # Check if we should use shared policy or individual policies per agent
    shared_policy = config['training'].get('shared_policy', True)  # Default to shared policy

    if shared_policy:
        # Compartimos la red neuronal para todos los agentes
        logger.info("Using shared policy for all agents")
        policy_learn = PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        policies = {agent: policy_learn
                    for agent in temp_env.agents}
    else:
        # Redes neuronales por agente
        logger.info("Using individual policies per agent")
        policies = {agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
                    for agent in temp_env.agents}

    temp_env.close()

    algorithm_name = config['training']['algorithm']
    config_algo = get_algorithm_config(config, env_config, policies)

    reward_function = config['training']['reward_function']

    # Create storage directory
    os.makedirs(config['storage_path'], exist_ok=True)

    tune.run(
        algorithm_name,
        config=config_algo.to_dict(),
        stop={"timesteps_total": config["training"]["timesteps_total"]},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
            num_to_keep=3,
            checkpoint_at_end=True,
            checkpoint_frequency=10
        ),
        storage_path=config["storage_path"],
        name=config["experiment_name"],
        resume=False,  # Always start fresh for batch training
        trial_name_creator=lambda trial: f"{trial.trainable_name}_{reward_function}_{trial.trial_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    logger.info(f"✅ Completed training: {config['experiment_name']}")
    logger.info(f"Models saved to: {config['storage_path']}")


def main():
    """Main function to run batch training"""
    parser = argparse.ArgumentParser(description="Batch train all reward functions with PPO and SAC")
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="Number of timesteps for each training (default: 5000)")
    parser.add_argument("--algorithms", nargs='+', choices=["PPO", "SAC"], default=["PPO", "SAC"],
                        help="Algorithms to train (default: both PPO and SAC)")
    parser.add_argument("--skip_pepe", action="store_true",
                        help="Skip training rewards_pepe.py classes")
    parser.add_argument("--skip_base", action="store_true",
                        help="Skip training BaseReward classes from rewards.py")
    parser.add_argument("--storage_path", type=str, default="../models_batch",
                        help="Base path for storing trained models (default: ../models_batch)")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue training other models if one fails")

    args = parser.parse_args()

    # Convert storage_path to absolute path to avoid URI issues
    args.storage_path = os.path.abspath(args.storage_path)

    # Initialize Ray
    init_ray()

    # Collect all reward functions to train
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

    # Calculate total training sessions
    total_trainings = len(reward_functions) * len(args.algorithms)
    completed_trainings = 0
    failed_trainings = []

    logger.info("=" * 70)
    logger.info("BATCH TRAINING SESSION STARTED")
    logger.info("=" * 70)
    logger.info(f"Total training sessions: {total_trainings}")
    logger.info(f"Timesteps per training: {args.timesteps}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"Reward functions: {[rf[0] for rf in reward_functions]}")
    logger.info(f"Models will be saved to: {args.storage_path}")
    logger.info("=" * 70)

    # Run training for each combination
    for algorithm in args.algorithms:
        for reward_func, source_file, base_class in reward_functions:
            training_id = f"{algorithm}_{reward_func}"
            logger.info(f"\n📊 Training {completed_trainings + 1}/{total_trainings}: {training_id}")
            logger.info(f"Source: {source_file} (inherits from {base_class})")

            try:
                # Create configuration for this training
                config = create_training_config(algorithm, reward_func, args.timesteps, args.storage_path)

                # Run training
                run_training_batch(config)

                completed_trainings += 1
                logger.info(f"✅ SUCCESS: {training_id}")

            except Exception as e:
                error_msg = f"❌ FAILED: {training_id} - {str(e)}"
                logger.error(error_msg)
                failed_trainings.append(training_id)

                if not args.continue_on_error:
                    logger.error("Stopping batch training due to error. Use --continue_on_error to skip failed trainings.")
                    break
                else:
                    logger.info("Continuing with next training...")
                    continue

        # Break outer loop if we stopped due to error
        if failed_trainings and not args.continue_on_error:
            break

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("BATCH TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total training sessions: {total_trainings}")
    logger.info(f"Completed successfully: {completed_trainings}")
    logger.info(f"Failed: {len(failed_trainings)}")

    if failed_trainings:
        logger.info("\nFailed trainings:")
        for failed_training in failed_trainings:
            logger.info(f"  - {failed_training}")
    else:
        logger.info("🎉 All trainings completed successfully!")

    logger.info(f"\nTrained models saved to: {args.storage_path}")
    logger.info("=" * 70)

    # Return appropriate exit code
    return 0 if len(failed_trainings) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
