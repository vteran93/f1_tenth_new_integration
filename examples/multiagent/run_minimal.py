"""
Minimal Training Script
Simplified version without NaN protection to isolate the segfault issue.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import original functionality
from run import (
    load_config, init_ray, get_logger, suppress_warnings,
    setup_experiment_config, find_experiment, create_env,
    get_algorithm_config
)
from ray import tune
from ray.tune.stopper import TrialPlateauStopper, CombinedStopper, Stopper
from ray.rllib.policy.policy import PolicySpec
from lib.callbacks import MultipleAgentCallbacks

suppress_warnings()
logger = get_logger(__name__)


class TimestepsStopper(Stopper):
    """Custom stopper that stops training after a certain number of timesteps."""
    
    def __init__(self, max_timesteps):
        self.max_timesteps = max_timesteps
    
    def __call__(self, trial_id, result):
        """Stop trial if timesteps_total exceeds max_timesteps."""
        return result.get("timesteps_total", 0) >= self.max_timesteps
    
    def stop_all(self):
        """Don't stop all experiments."""
        return False


def run_minimal_training(config):
    """Run training with minimal configuration to isolate issues."""
    
    logger.info("Starting minimal training (no NaN protection)...")
    
    # Create environment
    temp_env, env_config = create_env(config)
    
    # Setup policies
    shared_policy = config['training'].get('shared_policy', True)
    
    def shared_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    def individual_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id
    
    if shared_policy:
        logger.info("Using shared policy")
        shared_policy_name = "shared_policy"
        policies = {
            shared_policy_name: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        }
        policy_mapping_fn = shared_policy_mapping_fn
    else:
        logger.info("Using individual policies per agent")
        policies = {
            agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
            for agent in temp_env.agents
        }
        policy_mapping_fn = individual_policy_mapping_fn
    
    temp_env.close()
    
    # Get algorithm configuration with reduced parallelism
    num_env_runners = config.get('num_env_runners', 4)  # Reduced from 8
    num_envs_per_env_runner = config.get('num_envs_per_env_runner', 1)  # Reduced from 2
    config_algo = get_algorithm_config(config, env_config, policies, policy_mapping_fn,
                                       num_env_runners, num_envs_per_env_runner)
    
    # Use simple callback
    config_algo = config_algo.callbacks(MultipleAgentCallbacks)
    
    # Reduce batch sizes to minimize memory usage
    config_algo = config_algo.training(
        train_batch_size=256,  # Reduced from 512
        minibatch_size=64      # Reduced from 128
    )
    
    # Add debugging safeguards
    config_algo = config_algo.debugging(
        seed=42,
        log_level="INFO"
    )
    
    # Get algorithm name
    algorithm_name = config['training']['algorithm']
    
    # Define simpler stoppers
    timesteps_stopper = TimestepsStopper(max_timesteps=50000)  # Very short run
    
    # Run with minimal checkpoint configuration
    results = tune.run(
        algorithm_name,
        config=config_algo.to_dict(),
        stop=timesteps_stopper,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_score_attribute="env_runners/episode_return_mean",
            checkpoint_score_order="max",
            num_to_keep=1,  # Keep only 1 checkpoint
            checkpoint_at_end=False,  # Don't checkpoint at end
            checkpoint_frequency=50   # Checkpoint less frequently
        ),
        storage_path=config["storage_path"],
        name=config["name"] + "_minimal",
        resume=False,
        max_failures=0,  # Must be 0 when fail_fast=True
        fail_fast=True
    )
    
    logger.info("Minimal training completed successfully!")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimal PPO Training')
    parser.add_argument('--config', type=Path, default=Path('configs/experiments.yaml'))
    parser.add_argument('--experiment', type=str, required=True)
    
    args = parser.parse_args()
    
    config_path = args.config.resolve()
    config_dir = config_path.parent
    config_data = load_config(config_path)
    experiments = config_data.get('experiments', [])
    
    # Find experiment
    experiment_config = find_experiment(experiments, args.experiment)
    if not experiment_config:
        logger.error(f"Experiment '{args.experiment}' not found!")
        exit(1)
    
    # Setup experiment config
    config = setup_experiment_config(experiment_config, config_dir)
    
    # Initialize Ray
    init_ray()
    
    # Run training
    try:
        results = run_minimal_training(config)
        logger.info("Training finished successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise