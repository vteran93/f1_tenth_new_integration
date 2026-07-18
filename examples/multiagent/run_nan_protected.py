


"""
Safe NaN-Protected Training Script
Uses safe NaN detection without aggressive monkey patching to prevent crashes.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import safe NaN protection
from lib.safe_nan_detection import init_safe_nan_protection, SafeNaNRecoveryCallback

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


class SafeNaNProtectedCallback(MultipleAgentCallbacks):
    """
    Safe enhanced callback with NaN detection and recovery.
    """
    
    def __init__(self):
        super().__init__()
        self.nan_recovery = SafeNaNRecoveryCallback()
        self.iteration_count = 0
        self.last_checkpoint = None
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Check for NaN in training results and attempt recovery."""
        self.iteration_count += 1
        
        # Use safe NaN recovery
        result = self.nan_recovery.on_train_result(algorithm, result)
        
        # Continue with parent callback functionality
        return super().on_train_result(algorithm=algorithm, result=result, **kwargs)


def run_nan_protected_training(config):
    """Run training with NaN protection enabled."""
    
    # Initialize safe NaN protection
    logger.info("Initializing safe NaN protection...")
    init_safe_nan_protection()
    
    # Create environment
    temp_env, env_config = create_env(config)
    
    # Setup policies
    shared_policy = config['training'].get('shared_policy', True)
    
    def shared_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    def individual_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id
    
    if shared_policy:
        logger.info("Using shared policy with NaN protection")
        shared_policy_name = "shared_policy"
        policies = {
            shared_policy_name: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
        }
        policy_mapping_fn = shared_policy_mapping_fn
    else:
        logger.info("Using individual policies per agent with NaN protection")
        policies = {
            agent: PolicySpec(None, temp_env.observation_space, temp_env.action_space, {})
            for agent in temp_env.agents
        }
        policy_mapping_fn = individual_policy_mapping_fn
    
    temp_env.close()
    
    # Get algorithm configuration
    num_env_runners = config.get('num_env_runners', 8)
    num_envs_per_env_runner = config.get('num_envs_per_env_runner', 2)
    config_algo = get_algorithm_config(config, env_config, policies, policy_mapping_fn,
                                       num_env_runners, num_envs_per_env_runner)
    
    # Replace callback with safe NaN-protected version
    config_algo = config_algo.callbacks(SafeNaNProtectedCallback)
    
    # Add additional safeguards
    config_algo = config_algo.debugging(
        seed=42,
        log_level="INFO"
    )
    
    # Get algorithm name
    algorithm_name = config['training']['algorithm']
    
    # Define stoppers
    plateau_stopper = TrialPlateauStopper(
        metric="env_runners/episode_return_mean",
        std=10.0,
        num_results=20,
        mode="max",
    )
    
    timesteps_stopper = TimestepsStopper(max_timesteps=config["training"]["timesteps_total"])
    combined_stopper = CombinedStopper(plateau_stopper, timesteps_stopper)
    
    logger.info(f"Starting NaN-protected training: {config['name']}")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Timesteps target: {config['training']['timesteps_total']:,}")
    logger.info("NaN protection features:")
    logger.info("  ✓ Action distribution NaN detection and replacement")
    logger.info("  ✓ Model parameter NaN detection and reset")
    logger.info("  ✓ Forward hook NaN detection")
    logger.info("  ✓ Emergency checkpoint saving")
    logger.info("  ✓ Automatic failure recovery")
    
    # Run training with Ray Tune (following original run.py pattern)
    results = tune.run(
        algorithm_name,
        config=config_algo.to_dict(),
        stop=combined_stopper,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_score_attribute="env_runners/episode_return_mean",
            checkpoint_score_order="max",
            num_to_keep=3,
            checkpoint_at_end=True,
            checkpoint_frequency=10
        ),
        storage_path=config["storage_path"],
        name=config["name"],
        resume="AUTO+ERRORED",
        max_failures=3,
        fail_fast=False
    )
    
    logger.info("Training completed successfully!")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NaN-Protected PPO Training')
    parser.add_argument('--config', type=Path, default=Path('configs/experiments.yaml'))
    parser.add_argument('--experiment', type=str, required=True)
    
    args = parser.parse_args()
    
    config_path = args.config.resolve()
    config_dir = config_path.parent
    config_data = load_config(config_path)
    experiments = config_data.get('experiments', [])
    
    num_cpus = config_data.get('num_cpus', 16)
    init_ray(num_cpus=num_cpus)
    
    experiment = find_experiment(experiments, args.experiment)
    cfg = setup_experiment_config(experiment, config_dir)
    
    # Run NaN-protected training
    run_nan_protected_training(cfg)
