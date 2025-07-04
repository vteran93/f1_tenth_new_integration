import os
import logging

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.base_env import BaseEnv
import numpy as np
from examples.multiagent.lib.utils import get_logger
from f1tenth_gym.envs.f110_env import F110Env
from ray.tune.callback import Callback
from typing import Dict, Optional, Union, Any
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.typing import EpisodeType

# Debug logger specifically for callbacks
callback_logger = get_logger(__name__ + ".callbacks")


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


class EpisodeDuration(RLlibCallback):
    """A custom RLlib callback to track episode duration."""

    def __init__(self):
        super().__init__()
        callback_logger.debug("EpisodeDuration callback initialized")

    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Store the episode start time."""
        import time
        episode_id = getattr(episode, 'episode_id', id(episode))
        user_data = getattr(episode, 'user_data')
        user_data[str(episode_id)] = {'start_time': time.time()}

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
        """Calculate and log episode duration."""
        import time
        episode_id = getattr(episode, 'episode_id', id(episode))
        my_user_data = getattr(episode, 'user_data', {})

        if str(episode_id) in my_user_data:
            episode_duration = time.time() - my_user_data[str(episode_id)]['start_time']
            custom_metrics = getattr(episode, "custom_metrics", None)
            if custom_metrics is not None:
                custom_metrics["episode_duration"] = float(episode_duration)
                callback_logger.info(f"Episode {episode_id} duration: {episode_duration:.2f}s")
            else:
                callback_logger.warning(f"Episode {episode_id} has no custom_metrics attribute")
        else:
            callback_logger.warning(f"Episode {episode_id} start time not found in user_data")


class LapTimeProxy(RLlibCallback):
    """A custom RLlib callback to measure lap completion time as a proxy for lap time."""

    def __init__(self):
        super().__init__()

    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize lap tracking data."""
        import time
        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data')
        user_data[episode_id] = {
            'lap_start_time': time.time(),
            'lap_completed': False,
            'lap_time': 0.0
        }
        callback_logger.info(f"Episode on_episode_start {episode_id} is registered for lap time tracking")

    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        import time
        """Check for lap completion during episode."""
        callback_logger.info("Verifiying Episode on_episode_step called")
        if base_env is None or env_index is None:
            return

        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data')
        custom_metrics = getattr(episode, "custom_metrics", {})

        callback_logger.info(f"Verifiying Episode on_episode_step {episode_id} for lap time tracking")

        if episode_id not in user_data:
            return
        callback_logger.info(f"Episode on_episode_step {episode_id} is registered for lap time tracking")

        # Get the sub-environment and unwrap to F110Env
        sub_env = base_env.get_sub_environments()[env_index]
        f110_env: Any = getattr(sub_env, 'env', sub_env)

        # Check if any agent completed a lap using the underlying F110Env
        if f110_env and hasattr(f110_env, 'toggle_list'):
            callback_logger.info(f"Episode on_episode_step {episode_id} f110_env toggle_list found")
            if not user_data[episode_id]['lap_completed']:
                # Check if any agent has completed a lap (toggle_list >= 4 indicates lap completion)
                if np.any(getattr(f110_env, 'toggle_list') >= 4):
                    lap_time = time.time() - user_data[episode_id]['lap_start_time']
                    callback_logger.info(f"Episode {episode_id} lap_time: {lap_time:.2f}s")
                    user_data[episode_id]['lap_time'] = lap_time
                    user_data[episode_id]['lap_completed'] = True
                    custom_metrics["complete_lap_time"] = lap_time
                else:
                    incomplete_lap_time = time.time() - user_data[episode_id]['lap_start_time']
                    callback_logger.info(f"Episode {episode_id} incomplete_lap_time: {incomplete_lap_time:.2f}s")
                    user_data[episode_id]['incomplete_lap_time'] = incomplete_lap_time
                    user_data[episode_id]['lap_completed'] = False
                    custom_metrics["incomplete_lap_time"] = incomplete_lap_time

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
        """Called at the end of each episode to log lap time metrics and clean up episode data.

        This callback method processes completed episodes by:
        - Extracting lap time information from episode user data
        - Logging lap completion status and duration
        - Adding lap time metrics to custom metrics for tracking
        - Cleaning up episode-specific data from user_data storage

        For completed laps, it records the 'lap_time_proxy' metric.
        For incomplete laps, it records the 'incomplete_lap_time_proxy' metric.

        Args:
            episode: The episode object containing episode data and metrics
            worker: Optional environment runner instance
            base_env: Optional base environment instance
            policies: Optional dictionary mapping policy IDs to policy objects
            env_index: Optional environment index for multi-environment setups
            **kwargs: Additional keyword arguments

        Returns:
            None

        Note:
            This method assumes episode user_data contains lap tracking information
            with keys 'lap_completed', 'lap_time', and 'incomplete_lap_time'.
        """
        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        custom_metrics = getattr(episode, "custom_metrics", None)
        user_data = getattr(episode, 'user_data')
        callback_logger.info(f"Clening Episode {episode_id} on on_episode_end")

        if custom_metrics is not None and episode_id in user_data:
            callback_logger.info(f"Clening Episode valid custom_metrics {episode_id} on on_episode_end")

            if user_data[episode_id]['lap_completed']:
                callback_logger.info(f"We got custom metrics for {episode_id} with lap_completed on on_episode_end")
                lap_time_proxy = float(user_data[episode_id]['lap_time'])
                callback_logger.info(f"Episode {episode_id} duration: {lap_time_proxy:.2f}s")
                custom_metrics["lap_time_proxy"] = lap_time_proxy
            else:
                incomplete_lap_time = user_data[episode_id]['incomplete_lap_time']
                callback_logger.info(
                    f"Episode {episode_id} on_episode_end incomplete_lap_time: {incomplete_lap_time:.2f}s")
                custom_metrics["incomplete_lap_time_proxy"] = incomplete_lap_time

            # Clean up
            del user_data[episode_id]


class CollisionStats(RLlibCallback):
    """A custom RLlib callback to track collision statistics."""
    # TODO Needs review to check is working

    def __init__(self):
        super().__init__()

    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize collision tracking."""
        import time
        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data', {})

        user_data[episode_id] = {"collisions": {
            'start_time': time.time(),
            'collision_times': {},
            'collision_recorded': set()
        }}

    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Track collisions as they happen."""
        if base_env is None or env_index is None:
            return

        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data')

        if episode_id not in user_data:
            return

        # Get the sub-environment and unwrap to F110Env
        multiagent_env = base_env.get_sub_environments()[env_index]
        # Use the multiagent wrapper's collision tracking
        if hasattr(multiagent_env, '_crashed_agents') and hasattr(multiagent_env, 'agents'):
            import time
            current_time = time.time() - user_data[episode_id]["collisions"]['start_time']

            for agent in getattr(multiagent_env, '_crashed_agents', set()):
                if agent not in user_data[episode_id]["collisions"]['collision_recorded']:
                    user_data[episode_id]["collisions"]['collision_times'][agent] = current_time
                    user_data[episode_id]["collisions"]['collision_recorded'].add(agent)

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
        """Log collision statistics."""
        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        custom_metrics = getattr(episode, "custom_metrics", {})
        user_data = getattr(episode, 'user_data')
        if custom_metrics is not None and episode_id in user_data:
            collision_times = user_data[episode_id]["collisions"]['collision_times']

            # Log per-agent collision times
            for agent, collision_time in collision_times.items():
                custom_metrics[f"collision_time/{agent}"] = float(collision_time)

            # Log total number of collisions
            custom_metrics["total_collisions"] = len(collision_times)
            # Clean up
            del user_data[episode_id]["collisions"]


class AverageSpeed(RLlibCallback):
    """A custom RLlib callback to calculate average speed for each agent."""
    # TODO NECEISTA VALIDAR QUE FUNCIONA LA VELOCIDAD

    def __init__(self):
        super().__init__()
        self._speed_data = {}

    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize speed tracking."""
        episode_id = getattr(episode, 'episode_id', id(episode))
        self._speed_data[episode_id] = {
            'speed_samples': {},
            'step_count': 0
        }

    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Collect speed samples during episode."""
        if base_env is None or env_index is None:
            return

        episode_id = getattr(episode, 'episode_id', id(episode))
        if episode_id not in self._speed_data:
            return

        # Get the sub-environment and unwrap to F110Env
        sub_env = base_env.get_sub_environments()[env_index]
        f110_env = getattr(sub_env, 'env', sub_env)

        # Use F110Env's velocity data from the underlying simulator
        underlying_env = getattr(f110_env, 'env', None)
        if underlying_env and hasattr(underlying_env, 'sim') and hasattr(underlying_env.sim, 'agent_velocities'):
            self._speed_data[episode_id]['step_count'] += 1

            # Get velocities from the simulator (linear velocity magnitude)
            velocities = underlying_env.sim.agent_velocities[:, 0]  # vx component

            for i, agent in enumerate(getattr(f110_env, 'agents', [])):
                if agent not in self._speed_data[episode_id]['speed_samples']:
                    self._speed_data[episode_id]['speed_samples'][agent] = []
                self._speed_data[episode_id]['speed_samples'][agent].append(float(np.abs(velocities[i])))

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
        """Calculate and log average speeds."""
        episode_id = getattr(episode, 'episode_id', id(episode))
        custom_metrics = getattr(episode, "custom_metrics", None)
        if custom_metrics is not None and episode_id in self._speed_data:
            speed_samples = self._speed_data[episode_id]['speed_samples']

            # Calculate average speed for each agent
            for agent, speeds in speed_samples.items():
                if speeds:  # Avoid division by zero
                    avg_speed = np.mean(speeds)
                    custom_metrics[f"avg_speed/{agent}"] = float(avg_speed)

            # Calculate overall average speed across all agents
            all_speeds = [speed for speeds in speed_samples.values() for speed in speeds]
            if all_speeds:
                custom_metrics["avg_speed_all"] = float(np.mean(all_speeds))
            # Clean up
            del self._speed_data[episode_id]


CALLBACKS = [
    # EpisodeDuration, # Fixed
    # LapProgress, # Ok
    # LapTimeProxy,# ok
    CollisionStats,
    # AverageSpeed,
]


class MultipleAgentCallbacks(RLlibCallback):
    """A custom RLlib callback to handle multiple agent environments."""

    def __init__(self):
        super().__init__()
        self._callback_instances = {}
        callback_logger.debug("MultipleAgentCallbacks initialized")

    def _get_callback_instance(self, callback_class, episode_id):
        """Get or create callback instance for this episode."""
        if episode_id not in self._callback_instances:
            self._callback_instances[episode_id] = {}

        callback_name = callback_class.__name__
        if callback_name not in self._callback_instances[episode_id]:
            self._callback_instances[episode_id][callback_name] = callback_class()
            callback_logger.debug(f"Created {callback_name} instance for episode {episode_id}")

        return self._callback_instances[episode_id][callback_name]

    def on_episode_start(
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
        iterates over all the callbacks and calls their on_episode_start method.
        """
        episode_id = getattr(episode, 'episode_id', id(episode))
        callback_logger.debug(f"MultipleAgentCallbacks.on_episode_start called for episode {episode_id}")
        for callback_class in CALLBACKS:
            try:
                callback = self._get_callback_instance(callback_class, episode_id)
                if hasattr(callback, 'on_episode_start'):
                    callback_logger.debug(f"Calling on_episode_start for {callback_class.__name__}")
                    callback.on_episode_start(
                        episode=episode,
                        worker=worker,
                        base_env=base_env,
                        policies=policies,
                        env_index=env_index,
                        **kwargs
                    )
            except Exception as e:
                callback_logger.error(f"Error in {callback_class.__name__}.on_episode_start: {e}")

    def on_episode_step(
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
        Iterates over all the callbacks and calls their on_episode_step method.
        """
        episode_id = getattr(episode, 'episode_id', id(episode))
        callback_logger.debug(f"MultipleAgentCallbacks.on_episode_step called for episode {episode_id}")

        for callback_class in CALLBACKS:
            try:
                callback = self._get_callback_instance(callback_class, episode_id)
                if hasattr(callback, 'on_episode_step'):
                    callback_logger.debug(f"Calling on_episode_step for {callback_class.__name__}")
                    callback.on_episode_step(
                        episode=episode,
                        worker=worker,
                        base_env=base_env,
                        policies=policies,
                        env_index=env_index,
                        **kwargs
                    )
            except Exception as e:
                callback_logger.error(f"Error in {callback_class.__name__}.on_episode_step: {e}")

    def on_episode_end(self, *,
                       episode: Union[EpisodeType, EpisodeV2],
                       worker: Optional["EnvRunner"] = None,
                       base_env: Optional[BaseEnv] = None,
                       policies: Optional[Dict[str, Policy]] = None,
                       env_index: Optional[int] = None,
                       **kwargs, ):
        """
        Iterates over all the callbacks and calls their on_episode_end method."""
        episode_id = getattr(episode, 'episode_id', id(episode))
        callback_logger.debug(f"MultipleAgentCallbacks.on_episode_end called for episode {episode_id}")

        for callback_class in CALLBACKS:
            try:
                callback = self._get_callback_instance(callback_class, episode_id)
                if hasattr(callback, 'on_episode_end'):
                    callback_logger.debug(f"Calling on_episode_end for {callback_class.__name__}")
                    callback.on_episode_end(
                        episode=episode,
                        worker=worker,
                        base_env=base_env,
                        policies=policies,
                        env_index=env_index,
                        **kwargs)
            except Exception as e:
                callback_logger.error(f"Error in {callback_class.__name__}.on_episode_end: {e}")

        # Clean up callback instances for this episode
        if episode_id in self._callback_instances:
            del self._callback_instances[episode_id]
            callback_logger.debug(f"Cleaned up callback instances for episode {episode_id}")


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
