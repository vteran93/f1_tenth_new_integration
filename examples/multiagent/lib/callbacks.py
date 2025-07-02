import yaml
import os
import logging
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.base_env import BaseEnv
import numpy as np
from examples.multiagent.lib.utils import get_logger
from f1tenth_gym.envs import F110Env
from ray.tune.callback import Callback
from typing import Dict, Optional, Union
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.typing import EpisodeType


class LapProgress(RLlibCallback):
    """A custom RLlib callback to calculate and log lap progress metrics."""

    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Called at the end of an episode (old API stack).

        Calculates lap progress for each agent and logs mean, max, and min progress
        to the episode's custom metrics.
        """
        if base_env is None or env_index is None:
            return

        # Get the sub-environment and unwrap to F110Env
        sub_env = base_env.get_sub_environments()[env_index]
        f110_env = getattr(sub_env, 'env', sub_env)  # Unwrap if wrapped

        # Calculate lap progress for each agent
        if hasattr(f110_env, 'num_agents') and hasattr(f110_env, 'track'):
            lap_progress = []
            for i in range(f110_env.num_agents):  # type: ignore
                current_s, _ = f110_env.track.centerline.spline.calc_arclength_inaccurate(  # type: ignore
                    f110_env.poses_x[i], f110_env.poses_y[i]  # type: ignore
                )
                lap_progress.append(current_s / f110_env.track.centerline.spline.s[-1])  # type: ignore

            # Log metrics (using getattr for safety)
            custom_metrics = getattr(episode, "custom_metrics", None)
            if custom_metrics is not None:
                # Per-agent metrics
                for i, progress in enumerate(lap_progress):
                    agent_id = f"agent_{i}"
                    custom_metrics[f"lap_progress/{agent_id}"] = float(progress)
                
                # Combined metric - mean lap progress across all agents  
                custom_metrics["lap_progress"] = float(np.mean(lap_progress))


class SaveConfig(Callback):
    def __init__(self, resolved_config):
        self.resolved_config = resolved_config

    def setup(self, stop=None, num_samples=None, total_num_samples=None, **info):
        """Called once at the very beginning of training."""
        import yaml
        
        # Save config to the experiment's storage path
        experiment_name = self.resolved_config["name"]
        storage_path = self.resolved_config["storage_path"]
        experiment_dir = os.path.join(storage_path, experiment_name)
        
        # Create experiment directory if it doesn't exist
        os.makedirs(experiment_dir, exist_ok=True)
        
        config_file_path = os.path.join(experiment_dir, f"{experiment_name}_config.yaml")
        
        with open(config_file_path, 'w') as f:
            yaml.dump(self.resolved_config, f, default_flow_style=False, indent=2)
        
        logger = get_logger(__name__)
        logger.info(f"Saved resolved experiment config to: {config_file_path}")