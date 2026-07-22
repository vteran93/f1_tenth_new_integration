#!/usr/bin/env python3
"""
WSL2 + Intel CPU Compatible Training Script
Applies specific workarounds for WSL2/Intel CPU issues with Ray and Abseil.
"""

import sys
import argparse
import os
from pathlib import Path
import logging

# WSL2 + Intel CPU workarounds
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent OpenMP conflicts in WSL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Prevent BLAS threading issues
os.environ["RAY_DEDUP_LOGS"] = "0"  # Reduce logging overhead in WSL
os.environ["RAY_BACKEND_LOG_LEVEL"] = "warning"  # Reduce backend logging
# Disable problematic CPU optimizations that cause issues in WSL
os.environ["RAY_DISABLE_ADVANCED_VECTORIZATION"] = "1"

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import original functionality
from run import (
    load_config, get_logger, suppress_warnings,
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


class WSLCompatibleTimestepsStopper(Stopper):
    """WSL-compatible stopper that stops training after fewer timesteps to avoid crashes."""
    
    def __init__(self, max_timesteps):
        self.max_timesteps = max_timesteps
    
    def __call__(self, trial_id, result):
        """Stop trial if timesteps_total exceeds max_timesteps."""
        return result.get("timesteps_total", 0) >= self.max_timesteps
    
    def stop_all(self):
        """Don't stop all experiments."""
        return False


class WSLCompatibleCallback(MultipleAgentCallbacks):
    """
    WSL2-compatible callback with reduced logging and conservative resource usage.
    """
    
    def __init__(self):
        super().__init__()
        self.nan_recovery = SafeNaNRecoveryCallback()
        self.iteration_count = 0
        
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Check for NaN in training results with reduced logging for WSL."""
        self.iteration_count += 1
        
        # Use safe NaN recovery
        result = self.nan_recovery.on_train_result(algorithm, result)
        
        # Log progress every 20 iterations (less frequent for WSL)
        if self.iteration_count % 20 == 0:
            episode_return = result.get('episode_reward_mean', 'N/A')
            episode_length = result.get('episode_len_mean', 'N/A') 
            timesteps = result.get('timesteps_total', 0)
            logger.info(f"WSL Training - Iteration {self.iteration_count}: Return={episode_return}, Length={episode_length}, Timesteps={timesteps}")
        
        # Continue with parent callback functionality
        return super().on_train_result(algorithm=algorithm, result=result, **kwargs)


def configure_ray_wsl():
    """Configure Ray with WSL2-specific optimizations."""
    ray_config = {
        "ignore_reinit_error": True,
        "num_cpus": 2,  # Conservative CPU allocation for WSL
        "num_gpus": 0,  # No GPU in WSL2
        "log_to_driver": False,  # Reduce logging overhead
        "include_dashboard": False,  # Disable dashboard in WSL
        "object_store_memory": 1000000000,  # 1GB object store (conservative for WSL)
    }
    
    logger.info(f"Ray configured for WSL2: {ray_config}")
    return ray_config


def run_wsl_compatible_training(config):
    """Run training with WSL2 + Intel CPU compatibility."""
    
    # Initialize safe NaN protections
    logger.info("Initializing WSL2-compatible training with NaN protection...")
    init_safe_nan_protection()
    
    # Create environment
    temp_env, env_config = create_env(config)
    
    # Setup policies - always use individual policies for WSL stability
    logger.info("Using individual policies per agent for WSL2 stability")
    policies = {
        agent: PolicySpec(
            policy_class=None,
            observation_space=temp_env.observation_space,
            action_space=temp_env.action_space,
            config={}
        )
        for agent in temp_env.agents
    }
    
    def individual_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id
    
    policy_mapping_fn = individual_policy_mapping_fn
    temp_env.close()
    
    # WSL2-compatible configuration: minimal workers
    num_rollout_workers = 1  # Single worker for WSL stability
    num_envs_per_worker = 1  # Single environment
    
    logger.info(f"WSL2 settings: {num_rollout_workers} worker, {num_envs_per_worker} env per worker")
    
    # Get algorithm configuration
    algorithm_name = config['training']['algorithm']
    algorithm_config = get_algorithm_config(
        config, env_config, policies, policy_mapping_fn,
        num_rollout_workers, num_envs_per_worker
    )
    
    # WSL2-compatible configuration overrides
    config_dict = algorithm_config.to_dict()
    config_dict.update({
        # Minimal workers for WSL2
        "num_workers": num_rollout_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "rollout_fragment_length": 200,  # Smaller fragments for WSL
        "batch_mode": "complete_episodes",
        "num_cpus_per_worker": 1,
        
        # Conservative training settings for WSL2
        "train_batch_size": 200,     # Smaller batch size
        "sgd_minibatch_size": 50,    # Very small minibatches
        "num_sgd_iter": 1,           # Single SGD iteration
        "lr": 3e-5,                  # Lower learning rate
        "lr_schedule": [[0, 3e-5], [10000000, 0.0]],
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_clip": 0.5,            # Conservative gradient clipping
        
        # Resource settings for WSL2
        "num_gpus": 0,
        "num_cpus_for_driver": 1,
        
        # Environment settings
        "disable_env_checking": True,
        
        # Minimal evaluation for WSL2
        "evaluation_interval": None,  # Disable evaluation
        
        # Debugging
        "seed": 42,
        "log_level": "WARNING",  # Reduce logging in WSL
        
        # WSL-compatible callback
        "callbacks": WSLCompatibleCallback,
    })
    
    # Very conservative stopper for WSL2 (avoid long runs that crash)
    stopper = WSLCompatibleTimestepsStopper(max_timesteps=20000)  # Much shorter runs
    
    # Models path
    models_path = Path(config["storage_path"])
    experiment_name = config["name"]
    
    logger.info("Starting WSL2-compatible training with minimal resources...")
    
    # Use Ray 2.46.0 tune.run API with WSL2 optimizations
    analysis = tune.run(
        algorithm_name,
        name=f"{experiment_name}_wsl2_compatible",
        config=config_dict,
        storage_path=str(models_path),  # Use storage_path instead of local_dir
        stop=stopper,
        checkpoint_freq=25,  # Frequent checkpoints for WSL2
        keep_checkpoints_num=2,  # Keep fewer checkpoints
        verbose=0,  # Minimal verbosity for WSL
        max_failures=1,  # Allow minimal failures
        fail_fast=True,  # Fail fast in WSL2
        resources_per_trial={"cpu": 2, "gpu": 0},  # Explicit resource limits
    )
    
    logger.info("WSL2-compatible training completed!")
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSL2 + Intel CPU Compatible PPO Training')
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
    
    # Configure Ray with WSL2 optimizations
    ray_config = configure_ray_wsl()
    
    # Initialize Ray with WSL2-specific configuration
    import ray
    try:
        ray.shutdown()  # Shutdown any existing Ray instance
    except:
        pass
    
    ray.init(**ray_config)
    logger.info("Ray initialized with WSL2-compatible settings")
    
    # Run training
    try:
        results = run_wsl_compatible_training(config)
        logger.info("WSL2 training finished successfully")
    except Exception as e:
        logger.error(f"WSL2 training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        ray.shutdown()