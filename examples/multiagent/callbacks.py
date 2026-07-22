"""
Callbacks para integrar métricas de f1tenth_benchmarks con Ray Tune.
Este módulo permite guardar datos de entrenamiento y evaluación en el formato
requerido por f1tenth_benchmarks para análisis posterior.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging
from ray.tune.callback import Callback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2

logger = logging.getLogger(__name__)


class BenchmarkDataCollectorCallback(DefaultCallbacks):
    """
    Callback de RLLib para recopilar datos durante la evaluación
    en el formato requerido por f1tenth_benchmarks.
    """

    def __init__(self):
        super().__init__()
        self.episode_data = {}
        self.episode_rewards = []
        self.episode_progresses = []

    def on_episode_start(self,
                         *,
                         worker: RolloutWorker,
                         base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: EpisodeV2,
                         env_index: int,
                         **kwargs):
        """Inicializar estructuras de datos al inicio del episodio."""
        episode_id = episode.episode_id
        self.episode_data[episode_id] = {
            'states': [],
            'actions': [],
            'rewards': [],
            'infos': [],
            'progresses': [],
            'lap_times': [],
            'cross_track_errors': []
        }

    def on_episode_step(self,
                        *,
                        worker: RolloutWorker,
                        base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: EpisodeV2,
                        env_index: int,
                        **kwargs):
        """Recopilar datos en cada paso del episodio."""
        episode_id = episode.episode_id

        if episode_id not in self.episode_data:
            return

        # Obtener el último paso
        last_obs = episode.last_observation_for()
        last_action = episode.last_action_for()
        last_reward = episode.last_reward_for()
        last_info = episode.last_info_for()

        # Procesar para cada agente
        for agent_id in last_obs.keys():
            if agent_id in last_action and agent_id in last_info:
                # Extraer estado del vehículo desde las observaciones
                action = last_action[agent_id]
                reward = last_reward.get(agent_id, 0.0)
                info = last_info.get(agent_id, {})

                # Construir el estado en el formato f1tenth_benchmarks
                # [x, y, theta, speed, steering, angular_vel, slip_angle]
                if 'pose_x' in info and 'pose_y' in info:
                    x, y = info['pose_x'], info['pose_y']
                    theta = info.get('pose_theta', 0.0)
                    speed = info.get('linear_vels_x', 0.0)
                    steering = action[0] if len(action) > 0 else 0.0
                    angular_vel = info.get('angular_vels_z', 0.0)
                    slip_angle = info.get('slip_angle', 0.0)

                    state = [x, y, theta, speed, steering, angular_vel, slip_angle]
                    self.episode_data[episode_id]['states'].append(state)
                    self.episode_data[episode_id]['actions'].append(action)
                    self.episode_data[episode_id]['rewards'].append(reward)

                    # Guardar métricas adicionales si están disponibles
                    if 'progress' in info:
                        self.episode_data[episode_id]['progresses'].append(info['progress'])
                    if 'lap_time' in info:
                        self.episode_data[episode_id]['lap_times'].append(info['lap_time'])
                    if 'cross_track_error' in info:
                        self.episode_data[episode_id]['cross_track_errors'].append(info['cross_track_error'])

    def on_episode_end(self,
                       *,
                       worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: EpisodeV2,
                       env_index: int,
                       **kwargs):
        """Guardar datos del episodio completado."""
        episode_id = episode.episode_id

        if episode_id not in self.episode_data:
            return

        data = self.episode_data[episode_id]

        if len(data['states']) > 0:
            # Calcular métricas del episodio
            episode_reward = sum(data['rewards'])
            episode_progress = max(data['progresses']) if data['progresses'] else 0.0

            self.episode_rewards.append(episode_reward)
            self.episode_progresses.append(episode_progress)

            # Log basic metrics
            logger.info(f"Episode {episode_id}: Reward={episode_reward:.2f}, Progress={episode_progress:.2f}")

        # Limpiar datos del episodio
        del self.episode_data[episode_id]


class BenchmarkMetricsSaver(Callback):
    """
    Callback de Ray Tune para guardar métricas en formato f1tenth_benchmarks
    al final de cada trial.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = config.get('name', 'unknown_experiment')
        self.base_log_dir = Path(config.get('storage_path', './ray_results'))

    def on_trial_complete(self, iteration: int, trials: List, trial, **info):
        """
        Se ejecuta cuando un trial se completa.
        Aquí ejecutamos la evaluación final y guardamos métricas.
        """
        try:
            logger.info(f"Processing metrics for trial: {trial.trial_id}")

            # Crear estructura de directorios compatible con f1tenth_benchmarks
            trial_log_dir = self._create_log_structure(trial)

            # Ejecutar evaluación final con guardado de datos
            self._run_final_evaluation(trial, trial_log_dir)

            # Procesar métricas usando scripts de f1tenth_benchmarks
            self._process_benchmark_metrics(trial_log_dir, trial.trial_id)

        except Exception as e:
            logger.error(f"Error processing metrics for trial {trial.trial_id}: {e}")

    def _create_log_structure(self, trial) -> Path:
        """Crear estructura de directorios compatible con f1tenth_benchmarks."""
        trial_id = trial.trial_id.replace('/', '_').replace('\\', '_')

        # Crear estructura: Logs/{ExperimentName}/RawData_{TrialID}/
        log_dir = self.base_log_dir / "Logs" / self.experiment_name / f"RawData_{trial_id}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Crear directorio para imágenes
        img_dir = self.base_log_dir / "Logs" / self.experiment_name / f"Images_{trial_id}"
        img_dir.mkdir(parents=True, exist_ok=True)

        return log_dir

    def _run_final_evaluation(self, trial, log_dir: Path):
        """Ejecutar evaluación final y guardar datos en formato .npy."""
        try:
            from ray.rllib.algorithms.algorithm import Algorithm

            # Cargar el mejor checkpoint
            best_checkpoint = trial.checkpoint.dir_or_data
            if not best_checkpoint:
                logger.warning(f"No checkpoint found for trial {trial.trial_id}")
                return

            algo = Algorithm.from_checkpoint(best_checkpoint)

            # Crear entorno de evaluación
            env_config = self.config.get('env', {}).copy()
            env_config['render_mode'] = None  # Sin renderizado para evaluación

            from lib.utils import get_reward_class
            reward_class = get_reward_class(self.config)
            env = reward_class(env_config=env_config)

            # Determinar política
            shared_policy = self.config.get('training', {}).get('shared_policy', True)

            # Ejecutar episodios de evaluación
            num_episodes = 5  # Número de episodios para evaluación
            for episode_num in range(num_episodes):
                self._run_evaluation_episode(algo, env, log_dir, episode_num, shared_policy)

            env.close()

        except Exception as e:
            logger.error(f"Error in final evaluation: {e}")

    def _run_evaluation_episode(self, algo, env, log_dir: Path, episode_num: int, shared_policy: bool):
        """Ejecutar un episodio de evaluación y guardar los datos."""
        episode_data = {
            'states': [],
            'actions': [],
            'track_progresses': [],
            'tracking_accuracy': [],
            'tracking_points': []
        }

        obs, info = env.reset()
        terminated = {"__all__": False}
        step_count = 0

        while not terminated["__all__"] and step_count < 1000:  # Límite de pasos
            actions = {}

            for agent_id, agent_obs in obs.items():
                policy_id = "shared_policy" if shared_policy else agent_id
                actions[agent_id] = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id=policy_id,
                    explore=False
                )

            obs, reward, terminated, truncated, info = env.step(actions)

            # Recopilar datos de cada agente
            for agent_id in actions.keys():
                if agent_id in info:
                    agent_info = info[agent_id]
                    action = actions[agent_id]

                    # Construir estado [x, y, theta, speed, steering, angular_vel, slip_angle]
                    if 'pose_x' in agent_info and 'pose_y' in agent_info:
                        state = [
                            agent_info.get('pose_x', 0.0),
                            agent_info.get('pose_y', 0.0),
                            agent_info.get('pose_theta', 0.0),
                            agent_info.get('linear_vels_x', 0.0),
                            action[0] if len(action) > 0 else 0.0,
                            agent_info.get('angular_vels_z', 0.0),
                            agent_info.get('slip_angle', 0.0)
                        ]

                        episode_data['states'].append(state)
                        episode_data['actions'].append(action)

                        # Métricas adicionales si están disponibles
                        if 'progress' in agent_info:
                            episode_data['track_progresses'].append(agent_info['progress'])
                        if 'cross_track_error' in agent_info:
                            episode_data['tracking_accuracy'].append(agent_info['cross_track_error'])

            step_count += 1

        # Guardar datos del episodio
        if len(episode_data['states']) > 0:
            self._save_episode_data(episode_data, log_dir, episode_num)

    def _save_episode_data(self, episode_data: Dict, log_dir: Path, episode_num: int):
        """Guardar datos del episodio en formato .npy."""
        try:
            # Convertir listas a arrays numpy
            states = np.array(episode_data['states'])
            actions = np.array(episode_data['actions'])

            # Crear array combinado en formato f1tenth_benchmarks: [states, actions, progress]
            if len(states) > 0 and len(actions) > 0:
                # Asegurar que tenemos el mismo número de estados y acciones
                min_len = min(len(states), len(actions))
                states = states[:min_len]
                actions = actions[:min_len]

                # Combinar datos: estados (7) + acciones (2) = 9 columnas mínimo
                combined_data = np.column_stack([states, actions])

                # Añadir progreso si está disponible
                if episode_data['track_progresses']:
                    progresses = np.array(episode_data['track_progresses'][:min_len])
                    combined_data = np.column_stack([combined_data, progresses])

                # Guardar como SimLog_map_episode.npy
                map_name = self.config.get('env', {}).get('map', 'unknown')
                filename = f"SimLog_{map_name}_{episode_num}.npy"
                filepath = log_dir / filename

                np.save(filepath, combined_data)
                logger.info(f"Saved episode data: {filepath}")

                # Guardar datos de tracking accuracy si están disponibles
                if episode_data['tracking_accuracy']:
                    accuracy_data = np.column_stack([
                        episode_data['track_progresses'][:min_len],
                        episode_data['tracking_accuracy'][:min_len],
                        np.zeros((min_len, 2))  # Placeholder para tracking points
                    ])

                    accuracy_filename = f"TrackingAccuracy_{map_name}_{episode_num}.npy"
                    accuracy_filepath = log_dir / accuracy_filename
                    np.save(accuracy_filepath, accuracy_data)

        except Exception as e:
            logger.error(f"Error saving episode data: {e}")

    def _process_benchmark_metrics(self, log_dir: Path, trial_id: str):
        """Process metrics using local f1tenth_benchmarks."""
        try:
            # Use local f1tenth_benchmarks module for analysis
            from f1tenth_benchmarks import MetricsCalculator

            # Process basic metrics calculation
            calculator = MetricsCalculator()
            logger.info(f"Metrics calculation available for {trial_id}")

        except ImportError as ie:
            logger.warning(f"Could not import local f1tenth_benchmarks modules: {ie}")
        except Exception as e:
            logger.error(f"Error running benchmark analysis: {e}")


def create_summary_report(storage_path: str, experiment_name: str):
    """
    Crear un reporte resumen consolidado de todos los trials del experimento.
    """
    try:
        base_path = Path(storage_path) / "Logs" / experiment_name

        if not base_path.exists():
            logger.warning(f"No logs found at {base_path}")
            return

        summary_data = []

        # Buscar todos los directorios RawData_*
        for raw_data_dir in base_path.glob("RawData_*"):
            trial_id = raw_data_dir.name.replace("RawData_", "")

            # Buscar archivos .npy en el directorio
            npy_files = list(raw_data_dir.glob("SimLog_*.npy"))

            for npy_file in npy_files:
                try:
                    # Cargar datos
                    data = np.load(npy_file)

                    if len(data) > 0:
                        # Extraer métricas básicas
                        filename_parts = npy_file.stem.split('_')
                        map_name = filename_parts[1] if len(filename_parts) > 1 else 'unknown'
                        episode_num = filename_parts[2] if len(filename_parts) > 2 else '0'

                        # Calcular métricas del episodio
                        states = data[:, :7]  # Primeras 7 columnas son estados
                        speeds = states[:, 3]  # Velocidad

                        avg_speed = np.mean(speeds)
                        max_speed = np.max(speeds)
                        episode_length = len(data)

                        # Progreso si está disponible
                        progress = 0.0
                        if data.shape[1] > 9:  # Si hay columna de progreso
                            progress = np.max(data[:, -1])

                        summary_data.append({
                            'TestID': trial_id,
                            'Vehicle': experiment_name,
                            'VehicleID': trial_id,
                            'MapName': map_name,
                            'Lap': int(episode_num),
                            'LapTime': episode_length * 0.01,  # Asumiendo 100Hz
                            'AvgSpeed': avg_speed,
                            'MaxSpeed': max_speed,
                            'Progress': progress,
                            'CompletionRate': 1.0 if progress > 0.95 else 0.0
                        })

                except Exception as e:
                    logger.error(f"Error processing {npy_file}: {e}")

        # Crear DataFrame y guardar
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_file = base_path.parent / f"Results_{experiment_name}.csv"
            df.to_csv(summary_file, index=False, float_format='%.4f')
            logger.info(f"Created summary report: {summary_file}")

            # También crear el Summary.csv global
            global_summary = base_path.parent.parent / "Summary.csv"

            # Agregar métricas promedio por trial
            summary_stats = df.groupby(['TestID', 'MapName']).agg({
                'LapTime': 'mean',
                'Progress': 'max',
                'CompletionRate': 'max',
                'AvgSpeed': 'mean'
            }).reset_index()

            summary_stats['Vehicle'] = experiment_name
            summary_stats['VehicleID'] = summary_stats['TestID']
            summary_stats.rename(columns={
                'LapTime': 'AvgTime',
                'Progress': 'AvgProgress'
            }, inplace=True)

            # Agregar o actualizar el archivo global
            if global_summary.exists():
                existing_df = pd.read_csv(global_summary)
                # Remover entradas existentes para este experimento
                existing_df = existing_df[existing_df['Vehicle'] != experiment_name]
                updated_df = pd.concat([existing_df, summary_stats], ignore_index=True)
            else:
                updated_df = summary_stats

            updated_df.to_csv(global_summary, index=False, float_format='%.4f')
            logger.info(f"Updated global summary: {global_summary}")

    except Exception as e:
        logger.error(f"Error creating summary report: {e}")
