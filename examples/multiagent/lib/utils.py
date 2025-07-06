import yaml
import ray
import logging
import os
import warnings
import importlib
from ray.tune.callback import Callback
from ray.tune.analysis import ExperimentAnalysis


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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
