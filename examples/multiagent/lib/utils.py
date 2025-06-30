import yaml
import ray
import logging
import os
import warnings
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
