"""
Ray 2.46.0 Compatible Training Script
Uses conservative resource settings and Ray 2.46.0 APIs to avoid segmentation faults.
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
    Ray 2.46.0 compatible callback with safe NaN detection and resource management.
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
            # Ray 2.46.0 compatible metric names
            episode_return = result.get('episode_reward_mean', 'N/A')
            episode_length = result.get('episode_len_mean', 'N/A') 
            timesteps = result.get('timesteps_total', 0)
            logger.info(f"Iteration {self.iteration_count}: Return={episode_return}, Length={episode_length}, Timesteps={timesteps}")
        
        # Continue with parent callback functionality
        return super().on_train_result(algorithm=algorithm, result=result, **kwargs)


def configure_ray_resources():
    """Configure Ray with conservative resource settings for stability."""
    ray_config = {
        "ignore_reinit_error": True,
        "num_cpus": 4,  # Moderate CPU allocation 
        "num_gpus": 0,  # No GPU usage
        "log_to_driver": False,  # Reduce logging overhead
        "include_dashboard": False,  # Disable dashboard for reduced overhead
    }
    
    logger.info(f"Ray initialized with conservative settings: {ray_config}")
    return ray_config


def run_robust_training(config):
    """Run training with Ray 2.46.0 compatible configuration."""
    
    # Initialize safe NaN protections
    logger.info("Initializing Ray 2.46.0 compatible training with NaN protection...")
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
    
    # Conservative distributed mode configuration
    num_rollout_workers = 2  # Conservative number of workers
    num_envs_per_worker = 1  # Single environment per worker
    
    logger.info(f"Using conservative settings: {num_rollout_workers} workers, {num_envs_per_worker} env per worker")
    
    # Get algorithm configuration (Ray 2.46.0 style)
    algorithm_name = config['training']['algorithm']
    algorithm_config = get_algorithm_config(
        config, env_config, policies, policy_mapping_fn,
        num_rollout_workers, num_envs_per_worker
    )
    
    # Ray 2.46.0 compatible configuration - just use the config directly
    config_dict = algorithm_config
    
    # Override with conservative settings for Ray 2.46.0
    config_dict.update({
        # Rollout settings (Ray 2.46.0 uses num_workers, not num_rollout_workers)
        "num_workers": num_rollout_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "rollout_fragment_length": "auto",
        "batch_mode": "complete_episodes",
        "num_cpus_per_worker": 1,
        
        # Training settings
        "train_batch_size": 256,     # Reduced batch size for stability
        "sgd_minibatch_size": 64,    # Ray 2.46.0 uses sgd_minibatch_size
        "num_sgd_iter": 2,           # Reduced SGD iterations
        "lr": 5e-5,                  # Conservative learning rate
        "lr_schedule": [[0, 5e-5], [30000000, 0.0]],
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_clip": 0.1,
        
        # Resource settings
        "num_gpus": 0,               # CPU only
        "num_cpus_for_driver": 1,    # Ray 2.46.0 uses num_cpus_for_driver
        
        # Environment settings
        "disable_env_checking": True,
        
        # Evaluation settings
        "evaluation_interval": 100,
        "evaluation_duration": 5,
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "explore": False,
        },
        
        # Debugging
        "seed": 42,
        "log_level": "INFO",
        
        # Callbacks
        "callbacks": RobustNaNProtectedCallback,
    })
    
    # Conservative stopper
    stopper = ConservativeTimestepsStopper(max_timesteps=100000)
    
    # Models path
    models_path = Path(config["storage_path"])
    experiment_name = config["name"]
    
    logger.info("Starting Ray 2.46.0 compatible training with conservative resource settings...")
    
    # Use Ray 2.46.0 tune.run API
    analysis = tune.run(
        algorithm_name,
        name=f"{experiment_name}_ray246_compatible",
        config=config_dict,
        local_dir=str(models_path),
        stop=stopper,
        checkpoint_freq=50,
        keep_checkpoints_num=3,
        verbose=1,
        max_failures=2,
        fail_fast=False,
    )
    
    logger.info("Ray 2.46.0 compatible training completed successfully!")
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ray 2.46.0 Compatible PPO Training')
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