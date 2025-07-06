import yaml
import ray
import logging
import os
import warnings
import importlib
from ray.tune.callback import Callback
from ray.tune.analysis import ExperimentAnalysis
from ray import tune
import re


class TuneLoader(yaml.SafeLoader):
    """Custom YAML loader that supports Ray Tune functions."""
    
    def construct_scalar(self, node):
        """Override to handle tune functions."""
        value = super().construct_scalar(node)
        
        # Check if it's a tune function
        if isinstance(value, str) and value.startswith('tune.'):
            return self._parse_tune_function(value)
        
        return value
    
    def _parse_tune_function(self, value):
        """Parse tune function calls like 'tune.loguniform(0.95, 0.999)' or 'tune.choice([0.1, 0.2, 0.3])'."""
        pattern = r'tune\.(\w+)\((.*)\)'
        match = re.match(pattern, value)
        
        if not match:
            raise ValueError(f"Invalid tune function format: {value}")
        
        func_name = match.group(1)
        args_str = match.group(2)
        
        # Get the tune function first
        if not hasattr(tune, func_name):
            raise ValueError(f"Unknown tune function: {func_name}")
        
        tune_func = getattr(tune, func_name)
        
        # Parse arguments - use eval to handle complex expressions like lists
        try:
            # This handles cases like: (0.95, 0.999) or ([0.1, 0.2, 0.3]) or (5, 15)
            if args_str.strip():
                # Wrap in tuple syntax for eval
                parsed_args = eval(f"({args_str})")
                # If eval returns a single value, make it a tuple
                if not isinstance(parsed_args, tuple):
                    parsed_args = (parsed_args,)
            else:
                parsed_args = ()
                
            return tune_func(*parsed_args)
        except Exception as e:
            raise ValueError(f"Failed to parse arguments for {func_name}: {args_str}. Error: {e}")


def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=TuneLoader)


def init_ray(local_mode=False):
    ray.init(local_mode=local_mode, ignore_reinit_error=True,  num_cpus=16,                   # fuerza a Ray a usar 16 hilos
             num_gpus=0)


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_best_checkpoint(analysis: ExperimentAnalysis):
    best_trial = analysis.get_best_trial(metric="env_runners/episode_return_mean", mode="max")
    if best_trial:
        return analysis.get_best_checkpoint(best_trial, metric="env_runners/episode_return_mean", mode="max")
    return None


def get_experiment_path(experiment_name, storage_path="ray_results"):
    return os.path.join(storage_path, experiment_name)


def suppress_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Suppress gymnasium dtype casting warnings
    warnings.filterwarnings("ignore", message=".*Box low's precision lowered by casting to float32.*")
    warnings.filterwarnings("ignore", message=".*Box high's precision lowered by casting to float32.*")
    warnings.filterwarnings("ignore", message=".*The obs returned by the.*method was expecting numpy array dtype.*")
    warnings.filterwarnings("ignore", message=".*is not within the observation space.*")
    # Suppress gymnasium/gym warnings in general
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    warnings.filterwarnings("ignore", category=UserWarning, module="gym")


def get_reward_class(config):
    reward_function_name = config['training']['reward_function']
    reward_module = importlib.import_module('lib.rewards')
    return getattr(reward_module, reward_function_name)


def setup_experiment_config(experiment, config_dir):
    """Setup experiment configuration with resolved paths."""
    config = experiment.copy()

    # Resolve relative paths
    if not os.path.isabs(config["storage_path"]):
        config["storage_path"] = os.path.abspath(os.path.join(config_dir, config["storage_path"]))

    return config


def find_experiment(experiments, experiment_name):
    """Find experiment by name and return it, or exit with error if not found."""
    experiment = next((e for e in experiments if e["name"] == experiment_name), None)
    if not experiment:
        available = [e["name"] for e in experiments]
        logger = get_logger(__name__)
        logger.error(f"Experiment '{experiment_name}' not found. Available: {available}")
        exit(1)
    return experiment


def has_tune_params(config):
    """Check if config contains any Ray Tune hyperparameter objects."""
    def check_nested(obj):
        if hasattr(obj, '__module__') and obj.__module__ == 'ray.tune.search_space':
            return True
        if isinstance(obj, dict):
            return any(check_nested(v) for v in obj.values())
        if isinstance(obj, list):
            return any(check_nested(v) for v in obj)
        return False
    
    return check_nested(config)


def validate_hyperparameter_config(config):
    """Validate hyperparameter tuning configuration and raise clear errors if invalid."""
    hyperparameter_tuning = config.get("hyperparameter_tuning", False)
    has_tune_functions = has_tune_params(config)
    
    # Check for conflicting configurations
    if has_tune_functions and not hyperparameter_tuning:
        raise ValueError(
            "Configuration error: Found hyperparameter tuning functions (tune.uniform, tune.choice, etc.) "
            "but 'hyperparameter_tuning' is set to false or missing. "
            "Either set 'hyperparameter_tuning: true' or remove tune functions from your config."
        )
    
    if hyperparameter_tuning and "num_samples" not in config:
        raise ValueError(
            "Configuration error: 'hyperparameter_tuning' is enabled but 'num_samples' is not defined. "
            "Please add 'num_samples: <number>' to your config to specify how many trials to run."
        )
