#!/usr/bin/env python3
"""
Quick Training Script for F1TENTH RL with Best Practices

This script provides a streamlined interface for training F1TENTH RL agents
with integrated benchmarking, evaluation, and monitoring.
"""

from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import ray
import f1tenth_gym
import gymnasium as gym
import sys
import os
from pathlib import Path
import argparse
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Use local f1tenth_benchmarks module
import f1tenth_benchmarks

# Import required libraries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_f1tenth_env(env_config: EnvContext):
    """Factory function to create F1TENTH environment."""
    config = {
        'map': env_config.get('map', 'oval_small'),
        'num_agents': env_config.get('num_agents', 2),
        'timestep': env_config.get('timestep', 0.01),
        'integrator': env_config.get('integrator', 'rk4'),
        'num_beams': env_config.get('num_beams', 1080),
        'render_mode': env_config.get('render_mode', None),
    }

    env = gym.make('f1tenth_gym:f1tenth-v0', config=config)
    return env


def setup_ppo_config(env_config: Dict[str, Any], training_config: Dict[str, Any]) -> PPOConfig:
    """Setup PPO algorithm configuration."""

    config = (PPOConfig()
              .environment(
                  env='f1tenth_env',
                  env_config=env_config,
                  normalize_actions=training_config.get('normalize_actions', True)
    )
        .framework('torch')
        .training(
                  gamma=training_config.get('gamma', 0.99),
                  lr=training_config.get('lr', 5e-5),
                  train_batch_size=training_config.get('train_batch_size', 4000),
                  minibatch_size=training_config.get('sgd_minibatch_size', 256),
                  num_sgd_iter=training_config.get('num_sgd_iter', 10),
                  lambda_=training_config.get('lambda', 0.95),
                  clip_param=training_config.get('clip_param', 0.2),
                  vf_loss_coeff=training_config.get('vf_loss_coeff', 1.0),
                  entropy_coeff=training_config.get('entropy_coeff', 0.01),
    )
        .env_runners(
                  num_env_runners=training_config.get('num_workers', 4),
                  num_envs_per_env_runner=training_config.get('num_envs_per_worker', 1),
                  rollout_fragment_length=training_config.get('rollout_fragment_length', 200)
    )
        .resources(
                  num_gpus=training_config.get('num_gpus', 1.0),
                  num_cpus_per_worker=training_config.get('num_cpus_per_worker', 1)
    )
        .evaluation(
                  evaluation_interval=training_config.get('eval_interval', 2000),
                  evaluation_duration=training_config.get('eval_episodes', 5),
                  evaluation_config={
                      'render_env': False,
                      'explore': False
                  }
    )
        .debugging(
                  log_level='INFO'
    ))

    return config


def setup_sac_config(env_config: Dict[str, Any], training_config: Dict[str, Any]) -> SACConfig:
    """Setup SAC algorithm configuration."""

    config = (SACConfig()
              .environment(
                  env='f1tenth_env',
                  env_config=env_config,
                  normalize_actions=training_config.get('normalize_actions', True)
    )
        .framework('torch')
        .training(
                  gamma=training_config.get('gamma', 0.99),
                  lr=training_config.get('lr', 3e-4),
                  train_batch_size=training_config.get('train_batch_size', 256),
                  tau=training_config.get('tau', 0.005),
                  target_entropy=training_config.get('target_entropy', 'auto'),
                  n_step=training_config.get('n_step', 1),
                  replay_buffer_config={
                      'type': 'MultiAgentReplayBuffer',
                      'capacity': training_config.get('buffer_size', 1000000)
                  }
    )
        .env_runners(
                  num_env_runners=training_config.get('num_workers', 4),
                  num_envs_per_env_runner=training_config.get('num_envs_per_worker', 1)
    )
        .resources(
                  num_gpus=training_config.get('num_gpus', 1.0),
                  num_cpus_per_worker=training_config.get('num_cpus_per_worker', 1)
    )
        .evaluation(
                  evaluation_interval=training_config.get('eval_interval', 2000),
                  evaluation_duration=training_config.get('eval_episodes', 5),
                  evaluation_config={
                      'render_env': False,
                      'explore': False
                  }
    ))

    return config


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration for quick starts."""

    return {
        'experiment': {
            'name': f'f1tenth_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'algorithm': 'PPO',
            'description': 'Default F1TENTH RL training configuration'
        },
        'environment': {
            'map': 'oval_small',
            'num_agents': 2,
            'timestep': 0.01,
            'integrator': 'rk4',
            'num_beams': 1080,
            'render_mode': None
        },
        'training': {
            'timesteps_total': 3_000_000,
            'eval_interval': 2000,
            'eval_episodes': 5,
            'checkpoint_interval': 1000,
            'num_workers': 4,
            'num_gpus': 1.0,
            'num_cpus_per_worker': 1,
            'gamma': 0.99,
            'lr': 5e-5,
            'train_batch_size': 4000,
            'normalize_actions': True
        },
        'output': {
            'storage_path': './models',
            'save_logs': True,
            'save_checkpoints': True
        }
    }


def train_model(config: Dict[str, Any]) -> str:
    """Train a model with the given configuration."""

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    tune.register_env('f1tenth_env', create_f1tenth_env)

    # Setup algorithm configuration
    algorithm_name = config['experiment']['algorithm'].upper()

    if algorithm_name == 'PPO':
        algo_config = setup_ppo_config(config['environment'], config['training'])
    elif algorithm_name == 'SAC':
        algo_config = setup_sac_config(config['environment'], config['training'])
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    # Create storage directory
    storage_path = Path(config['output']['storage_path'])
    storage_path.mkdir(exist_ok=True, parents=True)

    # Setup training
    experiment_name = config['experiment']['name']

    logger.info(f"Starting training: {experiment_name}")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {config['environment']['map']}")
    logger.info(f"Total timesteps: {config['training']['timesteps_total']:,}")

    # Create and train algorithm
    algorithm = algo_config.build()

    try:
        # Training loop
        timesteps_trained = 0
        checkpoint_path = None

        while timesteps_trained < config['training']['timesteps_total']:

            # Train for one iteration
            result = algorithm.train()
            timesteps_trained = result['timesteps_total']

            # Log progress
            if result['training_iteration'] % 10 == 0:
                logger.info(
                    f"Iteration {result['training_iteration']:4d} | "
                    f"Timesteps: {timesteps_trained:8,} | "
                    f"Reward: {result.get('episode_reward_mean', 0.0):8.2f} | "
                    f"Episode Length: {result.get('episode_len_mean', 0.0):6.1f}"
                )

            # Save checkpoint
            if (config['output']['save_checkpoints'] and
                    result['training_iteration'] % (config['training']['checkpoint_interval'] // 100) == 0):

                checkpoint_path = algorithm.save(storage_path / experiment_name)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Final checkpoint
        if config['output']['save_checkpoints']:
            final_checkpoint = algorithm.save(storage_path / experiment_name)
            logger.info(f"Final checkpoint saved: {final_checkpoint}")
            checkpoint_path = final_checkpoint

        logger.info("Training completed successfully!")
        return checkpoint_path

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        algorithm.stop()
        ray.shutdown()


def quick_train(algorithm: str = 'PPO',
                map_name: str = 'oval_small',
                timesteps: int = 1_000_000,
                num_agents: int = 2) -> str:
    """Quick training function with minimal configuration."""

    config = create_default_config()

    # Update with provided parameters
    config['experiment']['algorithm'] = algorithm.upper()
    config['environment']['map'] = map_name
    config['environment']['num_agents'] = num_agents
    config['training']['timesteps_total'] = timesteps

    # Reduce workers for quick training
    config['training']['num_workers'] = 2
    config['training']['eval_interval'] = 5000

    logger.info("Starting quick training with simplified configuration")
    return train_model(config)


def main():
    """Main training script with command line interface."""

    parser = argparse.ArgumentParser(description='F1TENTH RL Training Script')

    # Training modes
    parser.add_argument('--mode', choices=['quick', 'config', 'interactive'],
                        default='quick', help='Training mode')

    # Quick training options
    parser.add_argument('--algorithm', choices=['PPO', 'SAC'], default='PPO',
                        help='RL algorithm to use')
    parser.add_argument('--map', default='oval_small',
                        help='Map to train on')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--agents', type=int, default=2,
                        help='Number of agents')

    # Configuration file
    parser.add_argument('--config', type=str,
                        help='Path to configuration YAML file')

    # Output options
    parser.add_argument('--output-dir', default='./models',
                        help='Output directory for models')
    parser.add_argument('--name', type=str,
                        help='Experiment name')

    args = parser.parse_args()

    try:
        if args.mode == 'quick':
            logger.info("Running in quick training mode")
            checkpoint_path = quick_train(
                algorithm=args.algorithm,
                map_name=args.map,
                timesteps=args.timesteps,
                num_agents=args.agents
            )

        elif args.mode == 'config':
            if not args.config:
                raise ValueError("Config file required for config mode")

            logger.info(f"Running with configuration file: {args.config}")
            config = load_experiment_config(args.config)

            # Override output directory if specified
            if args.output_dir:
                config['output']['storage_path'] = args.output_dir

            # Override experiment name if specified
            if args.name:
                config['experiment']['name'] = args.name

            checkpoint_path = train_model(config)

        elif args.mode == 'interactive':
            logger.info("Running in interactive mode")
            print("🏎️  F1TENTH RL Training - Interactive Mode")
            print("=" * 50)

            # Get user preferences
            algorithm = input("Algorithm (PPO/SAC) [PPO]: ").strip() or 'PPO'
            map_name = input("Map (oval_small/oval_large) [oval_small]: ").strip() or 'oval_small'
            timesteps = int(input("Training timesteps [1000000]: ").strip() or '1000000')
            num_agents = int(input("Number of agents [2]: ").strip() or '2')

            checkpoint_path = quick_train(
                algorithm=algorithm,
                map_name=map_name,
                timesteps=timesteps,
                num_agents=num_agents
            )

        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved at: {checkpoint_path}")
        print(f"🔍 You can now evaluate your model using the evaluation tools.")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\n⏹️  Training stopped by user")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
