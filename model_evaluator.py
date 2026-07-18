#!/usr/bin/env python3
"""
Advanced Model Evaluation System for F1TENTH RL

This module provides comprehensive evaluation capabilities including:
- Model performance benchmarking
- Cross-track evaluation
- Robustness testing
- Comparative analysis
- Automated reporting
"""

from ray.rllib.policy import Policy
from ray.rllib.algorithms import Algorithm as RLLibAlgorithm
import ray
import f1tenth_gym
import gymnasium as gym
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
import time
from dataclasses import dataclass, field
from datetime import datetime

# Use local f1tenth_benchmarks module
import f1tenth_benchmarks

# Import F1TENTH and RL libraries

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation."""

    # Performance metrics
    lap_times: List[float] = field(default_factory=list)
    completion_rates: List[float] = field(default_factory=list)
    progress_rates: List[float] = field(default_factory=list)
    collision_counts: List[int] = field(default_factory=list)

    # Driving behavior metrics
    avg_speeds: List[float] = field(default_factory=list)
    speed_variances: List[float] = field(default_factory=list)
    steering_smoothness: List[float] = field(default_factory=list)

    # Efficiency metrics
    path_efficiency: List[float] = field(default_factory=list)
    energy_consumption: List[float] = field(default_factory=list)

    # Robustness metrics
    noise_tolerance: Optional[float] = None
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'lap_times': {
                'mean': np.mean(self.lap_times) if self.lap_times else 0,
                'std': np.std(self.lap_times) if self.lap_times else 0,
                'min': np.min(self.lap_times) if self.lap_times else 0,
                'max': np.max(self.lap_times) if self.lap_times else 0,
                'count': len(self.lap_times)
            },
            'completion_rate': np.mean(self.completion_rates) if self.completion_rates else 0,
            'avg_progress_rate': np.mean(self.progress_rates) if self.progress_rates else 0,
            'total_collisions': sum(self.collision_counts),
            'avg_speed': np.mean(self.avg_speeds) if self.avg_speeds else 0,
            'steering_smoothness': np.mean(self.steering_smoothness) if self.steering_smoothness else 0,
            'path_efficiency': np.mean(self.path_efficiency) if self.path_efficiency else 0,
            'noise_tolerance': self.noise_tolerance,
            'parameter_sensitivity': self.parameter_sensitivity
        }


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Test episodes
    num_episodes: int = 100
    max_episode_steps: int = 1000

    # Test environments
    test_maps: List[str] = field(default_factory=lambda: ["oval_small", "oval_large"])

    # Robustness tests
    test_noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.1])
    test_parameter_variations: Dict[str, List[float]] = field(default_factory=dict)

    # Evaluation options
    render_evaluation: bool = False
    save_trajectories: bool = True
    record_videos: bool = False

    # Output settings
    output_dir: Path = Path("./evaluation_results")
    save_detailed_logs: bool = True


class ModelEvaluator:
    """Comprehensive model evaluation system."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.config.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize results storage
        self.results = {}
        self.detailed_logs = []

    def load_model(self, model_path: Union[str, Path]) -> RLLibAlgorithm:
        """Load a trained RLLib model."""
        try:
            # Initialize Ray if not already done
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            # Load the algorithm
            algorithm = RLLibAlgorithm.from_checkpoint(str(model_path))
            logger.info(f"Successfully loaded model from {model_path}")
            return algorithm

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def create_environment(self, map_name: str, **kwargs) -> gym.Env:
        """Create F1TENTH environment for evaluation."""
        try:
            env_config = {
                'map': map_name,
                'num_agents': kwargs.get('num_agents', 1),
                'timestep': kwargs.get('timestep', 0.01),
                'integrator': kwargs.get('integrator', 'rk4'),
                'num_beams': kwargs.get('num_beams', 1080),
                'render_mode': 'human' if self.config.render_evaluation else None
            }

            env = gym.make('f1tenth_gym:f1tenth-v0', config=env_config)
            logger.info(f"Created environment with map: {map_name}")
            return env

        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            raise

    def evaluate_single_episode(self,
                                algorithm: RLLibAlgorithm,
                                env: gym.Env,
                                episode_id: int,
                                noise_level: float = 0.0) -> Dict[str, Any]:
        """Evaluate a single episode and collect metrics."""

        obs, info = env.reset()
        done = False
        step = 0

        # Episode tracking
        episode_data = {
            'episode_id': episode_id,
            'noise_level': noise_level,
            'states': [],
            'actions': [],
            'rewards': [],
            'lap_times': [],
            'collisions': 0,
            'completion': False,
            'total_progress': 0.0,
            'total_reward': 0.0
        }

        speeds = []
        steering_angles = []
        positions = []

        while not done and step < self.config.max_episode_steps:
            # Get action from policy
            if isinstance(obs, dict):
                # Handle dict observations (multi-agent)
                agent_obs = obs
                action = {}
                for agent_id, agent_obs_val in agent_obs.items():
                    if agent_id == 'ego_idx':
                        continue
                    # Add noise if specified
                    if noise_level > 0:
                        if isinstance(agent_obs_val, np.ndarray):
                            noise = np.random.normal(0, noise_level, agent_obs_val.shape)
                            agent_obs_val = agent_obs_val + noise

                    action[agent_id] = algorithm.compute_single_action(agent_obs_val)
            else:
                # Handle array observations (single agent)
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, obs.shape)
                    obs = obs + noise

                action = algorithm.compute_single_action(obs)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Collect data
            episode_data['states'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)

            if isinstance(reward, dict):
                episode_data['total_reward'] += sum(reward.values())
            else:
                episode_data['total_reward'] += reward

            # Extract detailed metrics from info
            if isinstance(info, dict):
                # Handle multi-agent info
                for agent_id, agent_info in info.items():
                    if isinstance(agent_info, dict):
                        if 'lap_time' in agent_info:
                            episode_data['lap_times'].append(agent_info['lap_time'])
                        if 'collision' in agent_info and agent_info['collision']:
                            episode_data['collisions'] += 1
                        if 'progress' in agent_info:
                            episode_data['total_progress'] = max(
                                episode_data['total_progress'],
                                agent_info['progress']
                            )

                        # Collect driving behavior data
                        if 'linear_vels_x' in agent_info:
                            speeds.append(agent_info['linear_vels_x'])
                        if 'poses_x' in agent_info and 'poses_y' in agent_info:
                            positions.append((agent_info['poses_x'], agent_info['poses_y']))

                        # Extract steering from action if available
                        if isinstance(action, dict) and agent_id in action:
                            if len(action[agent_id]) >= 2:
                                steering_angles.append(action[agent_id][1])

            step += 1

        # Calculate derived metrics
        episode_data['completion'] = step < self.config.max_episode_steps and done
        episode_data['avg_speed'] = np.mean(speeds) if speeds else 0.0
        episode_data['speed_variance'] = np.var(speeds) if speeds else 0.0
        episode_data['steering_smoothness'] = self._calculate_steering_smoothness(steering_angles)
        episode_data['path_efficiency'] = self._calculate_path_efficiency(positions)

        return episode_data

    def _calculate_steering_smoothness(self, steering_angles: List[float]) -> float:
        """Calculate steering smoothness metric."""
        if len(steering_angles) < 2:
            return 0.0

        # Calculate smoothness as inverse of variance in steering changes
        steering_changes = np.diff(steering_angles)
        smoothness = 1.0 / (1.0 + np.var(steering_changes))
        return smoothness

    def _calculate_path_efficiency(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate path efficiency (actual path vs optimal path)."""
        if len(positions) < 2:
            return 0.0

        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)

        # Calculate straight-line distance (as approximation of optimal)
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[0][0]
            dy = positions[-1][1] - positions[0][1]
            straight_distance = np.sqrt(dx*dx + dy*dy)

            if total_distance > 0:
                efficiency = straight_distance / total_distance
                return min(efficiency, 1.0)  # Cap at 1.0

        return 0.0

    def evaluate_model_on_map(self,
                              algorithm: RLLibAlgorithm,
                              map_name: str,
                              noise_level: float = 0.0) -> EvaluationMetrics:
        """Evaluate model on a specific map."""

        logger.info(f"Evaluating on map: {map_name} (noise: {noise_level})")

        # Create environment
        env = self.create_environment(map_name)
        metrics = EvaluationMetrics()

        try:
            for episode in range(self.config.num_episodes):
                episode_data = self.evaluate_single_episode(
                    algorithm, env, episode, noise_level
                )

                # Extract metrics
                if episode_data['lap_times']:
                    metrics.lap_times.extend(episode_data['lap_times'])

                metrics.completion_rates.append(float(episode_data['completion']))
                metrics.progress_rates.append(episode_data['total_progress'])
                metrics.collision_counts.append(episode_data['collisions'])
                metrics.avg_speeds.append(episode_data['avg_speed'])
                metrics.speed_variances.append(episode_data['speed_variance'])
                metrics.steering_smoothness.append(episode_data['steering_smoothness'])
                metrics.path_efficiency.append(episode_data['path_efficiency'])

                # Store detailed logs if requested
                if self.config.save_detailed_logs:
                    self.detailed_logs.append({
                        'map': map_name,
                        'noise_level': noise_level,
                        **episode_data
                    })

                if (episode + 1) % 10 == 0:
                    logger.info(f"Completed {episode + 1}/{self.config.num_episodes} episodes")

        finally:
            env.close()

        return metrics

    def evaluate_robustness(self, algorithm: RLLibAlgorithm, map_name: str) -> Dict[str, float]:
        """Test model robustness to noise and parameter variations."""

        logger.info(f"Testing robustness on map: {map_name}")
        robustness_results = {}

        # Test noise tolerance
        baseline_metrics = self.evaluate_model_on_map(algorithm, map_name, noise_level=0.0)
        baseline_performance = np.mean(baseline_metrics.completion_rates)

        noise_tolerance = 0.0
        for noise_level in self.config.test_noise_levels[1:]:  # Skip 0.0
            noisy_metrics = self.evaluate_model_on_map(algorithm, map_name, noise_level)
            noisy_performance = np.mean(noisy_metrics.completion_rates)

            # Define tolerance as noise level where performance drops below 80% of baseline
            if noisy_performance >= 0.8 * baseline_performance:
                noise_tolerance = noise_level
            else:
                break

        robustness_results['noise_tolerance'] = noise_tolerance

        return robustness_results

    def evaluate_full_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Complete evaluation of a model across all test conditions."""

        logger.info(f"Starting full evaluation of model: {model_path}")
        start_time = datetime.now()

        # Load model
        algorithm = self.load_model(model_path)

        evaluation_results = {
            'model_path': str(model_path),
            'evaluation_config': self.config.__dict__,
            'start_time': start_time.isoformat(),
            'map_results': {},
            'robustness_results': {},
            'summary': {}
        }

        # Evaluate on each map
        all_metrics = []
        for map_name in self.config.test_maps:
            map_metrics = self.evaluate_model_on_map(algorithm, map_name)
            evaluation_results['map_results'][map_name] = map_metrics.to_dict()
            all_metrics.append(map_metrics)

            # Test robustness on this map
            robustness = self.evaluate_robustness(algorithm, map_name)
            evaluation_results['robustness_results'][map_name] = robustness

        # Calculate summary statistics
        all_lap_times = []
        all_completion_rates = []
        all_collision_counts = []

        for metrics in all_metrics:
            all_lap_times.extend(metrics.lap_times)
            all_completion_rates.extend(metrics.completion_rates)
            all_collision_counts.extend(metrics.collision_counts)

        evaluation_results['summary'] = {
            'total_episodes': len(all_completion_rates),
            'overall_completion_rate': np.mean(all_completion_rates),
            'overall_collision_rate': np.mean(all_collision_counts),
            'best_lap_time': np.min(all_lap_times) if all_lap_times else None,
            'avg_lap_time': np.mean(all_lap_times) if all_lap_times else None,
            'evaluation_duration': (datetime.now() - start_time).total_seconds()
        }

        # Save results
        results_file = self.config.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        logger.info(f"Evaluation complete. Results saved to {results_file}")

        return evaluation_results

    def compare_models(self, model_paths: List[Union[str, Path]]) -> pd.DataFrame:
        """Compare multiple models and generate comparison report."""

        logger.info(f"Comparing {len(model_paths)} models")

        comparison_data = []

        for model_path in model_paths:
            try:
                results = self.evaluate_full_model(model_path)

                model_summary = {
                    'model_name': Path(model_path).name,
                    'model_path': str(model_path),
                    **results['summary']
                }

                # Add map-specific results
                for map_name, map_results in results['map_results'].items():
                    model_summary[f'{map_name}_completion_rate'] = map_results['completion_rate']
                    model_summary[f'{map_name}_avg_lap_time'] = map_results['lap_times']['mean']
                    model_summary[f'{map_name}_collisions'] = map_results['total_collisions']

                comparison_data.append(model_summary)

            except Exception as e:
                logger.error(f"Failed to evaluate model {model_path}: {e}")
                continue

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Save comparison results
        comparison_file = self.config.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(comparison_file, index=False)

        logger.info(f"Model comparison saved to {comparison_file}")

        return comparison_df

    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""

        report = f"""
# F1TENTH Model Evaluation Report

**Model:** {results['model_path']}
**Evaluation Date:** {results['start_time']}
**Duration:** {results['summary']['evaluation_duration']:.2f} seconds

## Summary Results

- **Overall Completion Rate:** {results['summary']['overall_completion_rate']:.2%}
- **Total Episodes:** {results['summary']['total_episodes']}
- **Best Lap Time:** {results['summary']['best_lap_time']:.3f}s
- **Average Lap Time:** {results['summary']['avg_lap_time']:.3f}s
- **Collision Rate:** {results['summary']['overall_collision_rate']:.2f}

## Map-Specific Results

"""

        for map_name, map_results in results['map_results'].items():
            report += f"""
### {map_name}

- **Completion Rate:** {map_results['completion_rate']:.2%}
- **Average Lap Time:** {map_results['lap_times']['mean']:.3f}s (±{map_results['lap_times']['std']:.3f}s)
- **Best Lap Time:** {map_results['lap_times']['min']:.3f}s
- **Total Collisions:** {map_results['total_collisions']}
- **Path Efficiency:** {map_results['path_efficiency']:.3f}
- **Steering Smoothness:** {map_results['steering_smoothness']:.3f}
"""

        report += """
## Robustness Analysis

"""

        for map_name, robustness in results['robustness_results'].items():
            report += f"""
### {map_name}

- **Noise Tolerance:** {robustness['noise_tolerance']:.3f}
"""

        return report


def main():
    """Example usage of the model evaluator."""

    # Configure evaluation
    config = EvaluationConfig(
        num_episodes=20,  # Reduced for testing
        test_maps=["oval_small"],
        test_noise_levels=[0.0, 0.01, 0.05],
        output_dir=Path("./evaluation_results"),
        render_evaluation=False
    )

    # Create evaluator
    evaluator = ModelEvaluator(config)

    # Example: evaluate a single model (replace with actual model path)
    model_path = "./examples/models/ppo_model"  # Replace with actual path

    if Path(model_path).exists():
        results = evaluator.evaluate_full_model(model_path)

        # Generate report
        report = evaluator.generate_evaluation_report(results)
        print(report)

        # Save report
        report_file = config.output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_file}")
    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first or update the model path")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
