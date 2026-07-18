"""
Modern Robust Training Script
Uses conservative resource settings and modern Ray APIs to avoid segmentation faults.
"""

import sys
import argparse
import os
from pathlib import Path
import logging

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import original functionality
from run import (
    load_config, init_ray, get_logger, suppress_warnings,
    setup_experiment_config, find_experiment, create_env,
    get_algorithm_config
)
from ray import tune
from ray.tune.stopper import Stopper
from ray.rllib.policy.policy import PolicySpec
from lib.callbacks import MultipleAgentCallbacks

# Import safe NaN protection
from lib.safe_nan_detection import init_safe_nan_protection, SafeNaNRecoveryCallback

suppress_warnings()
logger = get_logger(__name__)


class ConservativeTimestepsStopper(Stopper):
    """Conservative stopper that stops training after a certain number of timesteps."""
    
    def __init__(self, max_timesteps):
        self.max_timesteps = max_timesteps
    
    def __call__(self, trial_id, result):
        """Stop trial if timesteps_total exceeds max_timesteps."""
        return result.get("timesteps_total", 0) >= self.max_timesteps
    
    def stop_all(self):
        """Don't stop all experiments."""
        return False


class RobustNaNProtectedCallback(MultipleAgentCallbacks):
    """
    Modern robust callback with safe NaN detection and resource management.
    """
    
    def __init__(self):
        super().__init__()
        self.nan_recovery = SafeNaNRecoveryCallback()
        self.iteration_count = 0
        
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Check for NaN in training results and attempt recovery."""
        self.iteration_count += 1
        
        # Use safe NaN recovery
        result = self.nan_recovery.on_train_result(algorithm, result)
        
        # Log progress every 10 iterations
        if self.iteration_count % 10 == 0:
            episode_return = result.get('env_runners/episode_return_mean', 'N/A')
            episode_length = result.get('env_runners/episode_len_mean', 'N/A')
            timesteps = result.get('timesteps_total', 0)
            logger.info(f"Iteration {self.iteration_count}: Return={episode_return}, Length={episode_length}, Timesteps={timesteps}")
        
        # Continue with parent callback functionality
        return super().on_train_result(algorithm=algorithm, result=result, **kwargs)


def configure_ray_resources():
    """Configure Ray with conservative resource settings for stability."""
    ray_config = {
        "ignore_reinit_error": True,
        "local_mode": False,  # Use distributed mode with conservative settings
        "num_cpus": 4,  # Moderate CPU allocation
        "num_gpus": 0,  # No GPU usage
        "log_to_driver": False,  # Reduce logging overhead
    }
    
    logger.info(f"Ray initialized with conservative settings: {ray_config}")
    return ray_config


def run_robust_training(config):
    """Run training with modern robust configuration."""
    
    # Initialize safe NaN protections
    logger.info("Initializing modern robust training with NaN protection...")
    init_safe_nan_protection()
    
    # Create environment
    temp_env, env_config = create_env(config)
    
    # Setup policies with conservative settings
    shared_policy = config['training'].get('shared_policy', False)  # Default to individual policies
    
    def shared_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    def individual_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id
    
    if shared_policy:
        logger.info("Using shared policy configuration")
        shared_policy_name = "shared_policy"
        policies = {
            shared_policy_name: PolicySpec(
                policy_class=None, 
                observation_space=temp_env.observation_space, 
                action_space=temp_env.action_space, 
                config={}
            )
        }
        policy_mapping_fn = shared_policy_mapping_fn
    else:
        logger.info("Using individual policies per agent (recommended for stability)")
        policies = {
            agent: PolicySpec(
                policy_class=None, 
                observation_space=temp_env.observation_space, 
                action_space=temp_env.action_space, 
                config={}
            )
            for agent in temp_env.agents
        }
        policy_mapping_fn = individual_policy_mapping_fn
    
    temp_env.close()
    
    # Local mode configuration (no distributed workers)
    num_env_runners = 0  # No separate env runners in local mode
    num_envs_per_env_runner = 1  # Single environment
    
    logger.info(f"Using local mode settings: {num_env_runners} workers, {num_envs_per_env_runner} env per worker")
    
    # Get algorithm configuration
    config_algo = get_algorithm_config(config, env_config, policies, policy_mapping_fn,
                                       num_env_runners, num_envs_per_env_runner)
    
    # Apply conservative training settings
    config_algo = config_algo.training(
        train_batch_size=256,      # Smaller batch size
        minibatch_size=64,         # Smaller minibatch size
        num_epochs=2,              # Keep reasonable epochs
        lr=5e-5,                   # Slightly lower learning rate for stability
        gamma=0.99,                # Standard discount factor
        lambda_=0.95,              # Standard GAE lambda
        clip_param=0.2,            # Standard PPO clip
        vf_loss_coeff=0.5,         # Reduce value function weight slightly
        entropy_coeff=0.01,        # Small entropy bonus
    )
    
    # Conservative resource settings
    config_algo = config_algo.resources(
        num_gpus=0,                # Use CPU only for stability
    ).learners(
        num_cpus_per_learner=1,    # One CPU per learner
        num_gpus_per_learner=0,    # No GPUs
    )
    
    # Conservative environment settings
    config_algo = config_algo.environment(
        disable_env_checking=True,  # Skip environment checks to reduce overhead
    )
    
    # Conservative env runner settings
    config_algo = config_algo.env_runners(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        rollout_fragment_length='auto',  # Let Ray decide optimal fragment length
        batch_mode='complete_episodes',  # Use complete episodes for stability
        num_cpus_per_env_runner=1,       # One CPU per env runner
    )
    
    # Conservative evaluation settings
    config_algo = config_algo.evaluation(
        evaluation_interval=100,    # Evaluate less frequently
        evaluation_duration=5,      # Shorter evaluation episodes
        evaluation_num_env_runners=1,  # Single evaluation worker
        evaluation_config={
            "explore": False,       # No exploration during evaluation
        }
    )
    
    # Use robust callback
    config_algo = config_algo.callbacks(RobustNaNProtectedCallback)
    
    # Conservative debugging settings
    config_algo = config_algo.debugging(
        seed=42,
        log_level="INFO",
        log_sys_usage=False,       # Reduce system monitoring overhead
    )
    
    # Get algorithm name
    algorithm_name = config['training']['algorithm']
    
    # Conservative stopper (shorter training for initial testing)
    timesteps_stopper = ConservativeTimestepsStopper(max_timesteps=100000)
    
    # Conservative checkpoint configuration
    checkpoint_config = tune.CheckpointConfig(
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_score_order="max",
        num_to_keep=2,              # Keep fewer checkpoints
        checkpoint_at_end=True,
        checkpoint_frequency=25     # Checkpoint less frequently
    )
    
    # Conservative run configuration
    run_config = tune.RunConfig(
        stop=timesteps_stopper,
        checkpoint_config=checkpoint_config,
        storage_path=config["storage_path"],
        name=config["name"] + "_robust",
        failure_config=tune.FailureConfig(
            max_failures=2,         # Allow some failures but not too many
            fail_fast=False,        # Don't fail fast, try to recover
        )
    )
    
    logger.info("Starting robust training with conservative resource settings...")
    
    # Run with conservative configuration
    tuner = tune.Tuner(
        algorithm_name,
        param_space=config_algo.to_dict(),
        run_config=run_config
    )
    
    results = tuner.fit()
    
    logger.info("Robust training completed successfully!")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modern Robust PPO Training')
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
    
    # Configure Ray with conservative settings
    ray_config = configure_ray_resources()
    
    # Initialize Ray with conservative configuration
    import ray
    try:
        ray.shutdown()  # Shutdown any existing Ray instance
    except:
        pass
    
    ray.init(**ray_config)
    logger.info(f"Ray initialized with conservative settings: {ray_config}")
    
    # Run training
    try:
        results = run_robust_training(config)
        logger.info("Training finished successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        ray.shutdown()