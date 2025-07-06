from scipy.interpolate import splev, splrep
import yaml
import ray
import logging
import os
import warnings
import importlib
from ray.tune.analysis import ExperimentAnalysis
import numpy as np
from numba import njit


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


def calculate_curvatures(points: np.ndarray) -> np.ndarray:
    """
    Calculate signed curvature for a sequence of points.
    Args:
        points: array of shape (N,2)
    Returns:
        curvatures: array of length N
    """
    x = points[:, 0]
    y = points[:, 1]
    t = np.arange(len(x), dtype=np.float64)
    spl_x = splrep(t, x.astype(np.float64))
    spl_y = splrep(t, y.astype(np.float64))
    dx = splev(t, spl_x, der=1)
    ddx = splev(t, spl_x, der=2)
    dy = splev(t, spl_y, der=1)
    ddy = splev(t, spl_y, der=2)
    denom = np.maximum((dx**2 + dy**2)**1.5, 1e-6)
    curv = (dx * ddy - dy * ddx) / denom
    return curv.astype(np.float32)


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """
    Encuentra el primer punto en la trayectoria que intersecta con un círculo.
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)

    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start).astype(np.float32)

        a = np.dot(V, V)
        b = np.float32(2.0) * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - np.float32(2.0) * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Calcula la actuación (velocidad y ángulo de dirección) para seguir un punto objetivo.
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle
