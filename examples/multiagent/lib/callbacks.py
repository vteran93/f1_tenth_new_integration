import os
import logging
import time

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
        user_data = getattr(episode, 'user_data', {})
        
        # Initialize episode data if not exists
        if str(episode_id) not in user_data:
            user_data[str(episode_id)] = {}
        
        # Initialize duration data
        user_data[str(episode_id)]['duration_data'] = {'start_time': time.time()}

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
        user_data = getattr(episode, 'user_data', {})
        custom_metrics = getattr(episode, "custom_metrics", None)

        if not custom_metrics:
            callback_logger.warning(f"Episode {episode_id} has no custom_metrics attribute")
            return

        if str(episode_id) not in user_data or 'duration_data' not in user_data[str(episode_id)]:
            callback_logger.warning(f"Episode {episode_id} duration data not found in user_data")
            return

        duration_data = user_data[str(episode_id)]['duration_data']
        episode_duration = time.time() - duration_data['start_time']
        custom_metrics["episode_duration"] = float(episode_duration)
        callback_logger.info(f"Episode {episode_id} duration: {episode_duration:.2f}s")

        # Clean up only duration data
        if 'duration_data' in user_data[str(episode_id)]:
            del user_data[str(episode_id)]['duration_data']


class LapTime(RLlibCallback):
    """Simplified callback to track lap times with minimal state."""

    def _initialize_lap_time_data(self, episode, base_env, env_index):
        """Initialize lap time tracking data if not already done."""
        if not (base_env and env_index is not None):
            return False

        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data', {})
        
        # Get F110Env
        sub_env = base_env.get_sub_environments()[env_index]
        f110_env = getattr(sub_env, 'env', sub_env)
        
        if hasattr(f110_env, 'num_agents'):
            # Initialize episode data if not exists
            if episode_id not in user_data:
                user_data[episode_id] = {}
            
            # Initialize lap time data if not exists
            if 'lap_time_data' not in user_data[episode_id]:
                callback_logger.debug(f"Initializing lap time tracking for episode {episode_id} with {f110_env.num_agents} agents")
                user_data[episode_id]['lap_time_data'] = {
                    'start_time': 0.0,
                    'agent_lap_times': [None] * f110_env.num_agents,
                    'last_toggle_list': [0] * f110_env.num_agents,
                    'first_step': True
                }
            return True
        return False

    def on_episode_start(self, *, episode, base_env=None, env_index=None, **kwargs) -> None:
        """Initialize lap time tracking."""
        if not self._initialize_lap_time_data(episode, base_env, env_index):
            callback_logger.warning("Failed to initialize lap time data in on_episode_start")
            return

    def on_episode_step(self, *, episode, base_env=None, env_index=None, **kwargs) -> None:
        """Track lap completion."""
        # Initialize data if not done yet (defensive programming)
        if not self._initialize_lap_time_data(episode, base_env, env_index):
            callback_logger.warning("Failed to initialize lap time data in on_episode_step")
            return

        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data', {})
        
        # Check for the existence of data for this episode
        if episode_id not in user_data or 'lap_time_data' not in user_data[episode_id]:
            callback_logger.warning(f"Episode {episode_id} has no lap_time_data, skipping on_episode_step")
            return

        # Get F110Env
        sub_env = base_env.get_sub_environments()[env_index]
        f110_env = getattr(sub_env, 'env', sub_env)
        
        if not (hasattr(f110_env, 'toggle_list') and hasattr(f110_env, 'current_time')):
            callback_logger.warning(f"F110Env in episode {episode_id} does not have toggle_list or current_time, skipping on_episode_step")
            return

        data = user_data[episode_id]['lap_time_data']
        
        # Set start time on first step
        if data['first_step']:
            callback_logger.debug(f"Setting start time for episode {episode_id} at {f110_env.current_time}")
            data['start_time'] = f110_env.current_time
            data['last_toggle_list'] = list(f110_env.toggle_list)
            data['first_step'] = False
            return

        # Check for lap completion
        for i in range(f110_env.num_agents):
            if (data['last_toggle_list'][i] < 4 and 
                f110_env.toggle_list[i] >= 4 and 
                data['agent_lap_times'][i] is None):
                callback_logger.debug(f"Agent {i} completed lap in episode {episode_id} at time {f110_env.current_time}")
                data['agent_lap_times'][i] = f110_env.current_time - data['start_time']

        data['last_toggle_list'] = list(f110_env.toggle_list)

    def on_episode_end(self, *, episode, base_env=None, env_index=None, **kwargs) -> None:
        """Log the exact metrics requested."""
        if not (base_env and env_index is not None):
            callback_logger.warning("Base environment or env_index is None, skipping on_episode_end")
            return

        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data', {})
        custom_metrics = getattr(episode, "custom_metrics", None)
        
        if not custom_metrics:
            callback_logger.warning(f"Episode {episode_id} has no custom_metrics, skipping on_episode_end")
            return
            
        if episode_id not in user_data or 'lap_time_data' not in user_data[episode_id]:
            callback_logger.warning(f"Episode {episode_id} has no lap_time_data, skipping on_episode_end")
            return

        # Get F110Env
        sub_env = base_env.get_sub_environments()[env_index]
        f110_env = getattr(sub_env, 'env', sub_env)
        
        if not hasattr(f110_env, 'current_time'):
            callback_logger.warning(f"F110Env in episode {episode_id} does not have current_time, skipping on_episode_end")
            return

        data = user_data[episode_id]['lap_time_data']
        completed_times = []
        
        # Per-agent metrics
        for i in range(f110_env.num_agents):
            if data['agent_lap_times'][i] is not None:
                callback_logger.debug(f"Agent {i} lap time in episode {episode_id}: {data['agent_lap_times'][i]}")
                # Agent completed lap
                custom_metrics[f"lap_time/agent_{i}"] = float(data['agent_lap_times'][i])
                completed_times.append(data['agent_lap_times'][i])
            else:
                callback_logger.debug(f"Agent {i} did not complete lap in episode {episode_id}")
                # Agent didn't complete lap
                incomplete_time = f110_env.current_time - data['start_time']
                custom_metrics[f"incomplete_lap_time/agent_{i}"] = float(incomplete_time)
        
        # Overall metrics
        if completed_times:
            callback_logger.debug(f"Overall lap time for episode {episode_id}: {np.mean(completed_times)}")
            custom_metrics["lap_time"] = float(np.mean(completed_times))
        else:
            callback_logger.debug(f"No agents completed laps in episode {episode_id}, using incomplete lap time")
            custom_metrics["incomplete_lap_time"] = float(f110_env.current_time - data['start_time'])

        # Clean up only lap time data, not the entire episode data
        if 'lap_time_data' in user_data[episode_id]:
            callback_logger.debug(f"Cleaning up lap_time_data for episode {episode_id}")
            del user_data[episode_id]['lap_time_data']


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

        # Initialize episode data if not exists
        if episode_id not in user_data:
            user_data[episode_id] = {}

        user_data[episode_id]["collision_data"] = {
            'start_time': time.time(),
            'collision_times': {},
            'collision_recorded': set()
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
        """Track collisions as they happen."""
        if base_env is None or env_index is None:
            return

        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data', {})

        if episode_id not in user_data or 'collision_data' not in user_data[episode_id]:
            return

        # Get the sub-environment and unwrap to F110Env
        multiagent_env = base_env.get_sub_environments()[env_index]
        # Use the multiagent wrapper's collision tracking
        if hasattr(multiagent_env, '_crashed_agents') and hasattr(multiagent_env, 'agents'):
            import time
            current_time = time.time() - user_data[episode_id]["collision_data"]['start_time']

            for agent in getattr(multiagent_env, '_crashed_agents', set()):
                if agent not in user_data[episode_id]["collision_data"]['collision_recorded']:
                    user_data[episode_id]["collision_data"]['collision_times'][agent] = current_time
                    user_data[episode_id]["collision_data"]['collision_recorded'].add(agent)

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
        custom_metrics = getattr(episode, "custom_metrics", None)
        user_data = getattr(episode, 'user_data', {})
        
        if not custom_metrics:
            return
            
        if episode_id not in user_data or 'collision_data' not in user_data[episode_id]:
            return
            
        collision_times = user_data[episode_id]["collision_data"]['collision_times']

        # Log per-agent collision times
        for agent, collision_time in collision_times.items():
            custom_metrics[f"collision_time/{agent}"] = float(collision_time)

        # Log total number of collisions
        custom_metrics["total_collisions"] = len(collision_times)
        
        # Clean up only collision data
        if 'collision_data' in user_data[episode_id]:
            del user_data[episode_id]['collision_data']


class AverageSpeed(RLlibCallback):
    """A custom RLlib callback to calculate average speed for each agent using incremental mean.
    Incremental mean is used to avoid storing all speed values in memory.
    """

    def on_episode_start(self, *, episode, **kwargs) -> None:
        """Initialize speed tracking for this episode."""
        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data', {})
        
        # Initialize episode data if not exists
        if episode_id not in user_data:
            user_data[episode_id] = {}
        
        # Initialize speed data
        user_data[episode_id]['speed_data'] = {}

    def on_episode_step(self, *, episode, base_env=None, env_index=None, **kwargs) -> None:
        """Update incremental speed average during episode."""
        if not (base_env and env_index is not None):
            return

        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data', {})
        
        # Initialize data if not exists (defensive programming)
        if episode_id not in user_data:
            user_data[episode_id] = {}
        if 'speed_data' not in user_data[episode_id]:
            user_data[episode_id]['speed_data'] = {}
        
        episode_data = user_data[episode_id]
        if not episode_data:
            return

        f110_env = getattr(base_env.get_sub_environments()[env_index], 'env', None)
        if not (f110_env and hasattr(f110_env, 'sim') and hasattr(f110_env.sim, 'agents')):
            return

        # Update incremental average for each agent
        speed_data = episode_data['speed_data']
        for i in range(f110_env.num_agents):
            agent_id = f"agent_{i}"
            speed = abs(f110_env.sim.agents[i].state[3])
            
            if agent_id not in speed_data:
                speed_data[agent_id] = {'avg': speed, 'count': 1}
            else:
                # Incremental mean: new_avg = old_avg + (new_value - old_avg) / new_count
                count = speed_data[agent_id]['count'] + 1
                speed_data[agent_id]['avg'] += (speed - speed_data[agent_id]['avg']) / count
                speed_data[agent_id]['count'] = count

    def on_episode_end(self, *, episode, **kwargs) -> None:
        """Log average speeds calculated incrementally."""
        episode_id = str(getattr(episode, 'episode_id', id(episode)))
        user_data = getattr(episode, 'user_data', {})
        custom_metrics = getattr(episode, "custom_metrics", None)

        if not custom_metrics:
            return
            
        if episode_id not in user_data or 'speed_data' not in user_data[episode_id]:
            return

        speed_data = user_data[episode_id]['speed_data']
        if speed_data:
            # Per-agent metrics
            avg_speeds = []
            for i, (agent_id, data) in enumerate(speed_data.items()):
                custom_metrics[f"avg_speed/agent_{i}"] = float(data['avg'])
                avg_speeds.append(data['avg'])
            
            # Combined metric
            custom_metrics["avg_speed"] = float(np.mean(avg_speeds))

        # Clean up only speed data, not the entire episode data
        if 'speed_data' in user_data[episode_id]:
            del user_data[episode_id]['speed_data']


CALLBACKS = [
    # EpisodeDuration, # Fixed
    LapProgress, # Ok
    LapTime,# ok
    # CollisionStats, #ok
    AverageSpeed,
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
        
        # Clean up any remaining episode data if the episode dict is empty
        user_data = getattr(episode, 'user_data', {})
        if str(episode_id) in user_data and not user_data[str(episode_id)]:
            del user_data[str(episode_id)]
            callback_logger.debug(f"Cleaned up empty episode data for episode {episode_id}")


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
